# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import math

# Third Party
import pytest

# First Party
from lmcache.v1.cache_controller.controllers.kv_controller import KVController
from lmcache.v1.cache_controller.message import (
    FullLookupMsg,
    KVAdmitMsg,
    KVEvictMsg,
    LookupMsg,
)
from tests.v1.utils import generate_tokens


@pytest.mark.parametrize("instance_id", ["test1", "test2"])
@pytest.mark.parametrize("worker_id", [0, 1])
@pytest.mark.parametrize("location", ["LocalCPUBackend", "LocalDiskBackend"])
@pytest.mark.asyncio
async def test_kv_chunk_lifecycle(instance_id, worker_id, location):
    kv_controller = KVController()
    """Test KV chunk lifecycle: add -> remove -> deregister"""

    # Generate tokens and get key from token_database
    tokens = generate_tokens(256, "cpu")  # Use chunk_size tokens

    # Get the first key from token_database
    for start, end, key in kv_controller.token_database.process_tokens(
        tokens, make_key=False
    ):
        assert isinstance(key, int)

        # Test admit
        admit_msg = KVAdmitMsg(
            instance_id=instance_id, worker_id=worker_id, key=key, location=location
        )
        await kv_controller.admit(admit_msg)

        # Verify KV chunk added
        assert str(key) in kv_controller.kv_pool
        assert len(kv_controller.kv_pool[str(key)]) == 1
        metadata = kv_controller.kv_pool[str(key)][0]
        assert metadata.instance_id == instance_id
        assert metadata.worker_id == worker_id
        assert metadata.location == location

        # Test evict
        evict_msg = KVEvictMsg(
            instance_id=instance_id, worker_id=worker_id, key=key, location=location
        )
        await kv_controller.evict(evict_msg)

        # Verify KV chunk removed
        assert key not in kv_controller.kv_pool

        # Test deregister on empty pool
        await kv_controller.deregister(instance_id, worker_id)
        assert len(kv_controller.kv_pool) == 0
        break  # Only test with first key


@pytest.mark.asyncio
async def test_kv_chunk_multiple_instances():
    kv_controller = KVController()
    """Test multiple instances KV chunk management"""

    # Generate tokens and get key from token_database
    tokens = generate_tokens(256, "cpu")  # Use chunk_size tokens

    # Get the first key from token_database
    for start, end, key in kv_controller.token_database.process_tokens(
        tokens, make_key=False
    ):
        # Add KV chunks for multiple instances
        instances = [
            ("instance1", 0, "LocalCPUBackend"),
            ("instance1", 1, "LocalDiskBackend"),
            ("instance2", 0, "LocalCPUBackend"),
        ]

        for instance_id, worker_id, location in instances:
            admit_msg = KVAdmitMsg(
                instance_id=instance_id, worker_id=worker_id, key=key, location=location
            )
            await kv_controller.admit(admit_msg)

        # Verify all KV chunks added
        assert str(key) in kv_controller.kv_pool
        assert len(kv_controller.kv_pool[str(key)]) == 3

        # Remove one instance KV chunk
        evict_msg = KVEvictMsg(
            instance_id="instance1", worker_id=0, key=key, location="LocalCPUBackend"
        )
        await kv_controller.evict(evict_msg)

        # Verify only 2 KV chunks remain
        assert len(kv_controller.kv_pool[str(key)]) == 2

        # Deregister all instance1 workers
        await kv_controller.deregister("instance1", 0)
        await kv_controller.deregister("instance1", 1)

        # Verify only instance2 KV chunk remains
        assert len(kv_controller.kv_pool[str(key)]) == 1
        assert kv_controller.kv_pool[str(key)][0].instance_id == "instance2"
        break  # Only test with first key


@pytest.mark.parametrize("token_length", [100, 500])
@pytest.mark.asyncio
async def test_lookup_functionality(token_length):
    kv_controller = KVController()
    """Test basic lookup functionality"""
    tokens = generate_tokens(token_length, "cpu")

    # Add KV chunks for each chunk using token_database to generate keys
    for start, end, key in kv_controller.token_database.process_tokens(
        tokens, make_key=False
    ):
        assert isinstance(key, int)
        admit_msg = KVAdmitMsg(
            instance_id="test_instance",
            worker_id=0,
            key=key,
            location="LocalCPUBackend",
        )
        await kv_controller.admit(admit_msg)

    # Test basic lookup
    lookup_msg = LookupMsg(event_id="test_lookup", tokens=tokens.tolist())
    result = await kv_controller.lookup(lookup_msg)

    # Verify result
    assert "test_instance" in result.layout_info
    location, end = result.layout_info["test_instance"]
    assert location == "LocalCPUBackend"
    assert end == token_length


@pytest.mark.parametrize("token_length", [100, 500])
@pytest.mark.asyncio
async def test_full_lookup_functionality(token_length):
    kv_controller = KVController()
    """Test full lookup functionality"""
    chunk_size = kv_controller.token_database.chunk_size

    tokens = generate_tokens(token_length, "cpu")
    expected_chunks = math.ceil(token_length / chunk_size)

    # Add KV chunks for each chunk using token_database to generate keys
    for start, end, key in kv_controller.token_database.process_tokens(
        tokens, make_key=False
    ):
        admit_msg = KVAdmitMsg(
            instance_id="test_instance",
            worker_id=0,
            key=key,
            location="LocalCPUBackend",
        )
        await kv_controller.admit(admit_msg)

    # Test full lookup
    full_lookup_msg = FullLookupMsg(event_id="test_full_lookup", tokens=tokens.tolist())
    full_result = await kv_controller.full_lookup(full_lookup_msg)

    # Verify full lookup result
    assert len(full_result.matched_info) == 1
    instance_id, cache_list = full_result.matched_info[0]
    assert instance_id == "test_instance"
    assert len(cache_list) == expected_chunks
    assert full_result.chunk_size == chunk_size


@pytest.mark.asyncio
async def test_lookup_edge_cases():
    kv_controller = KVController()
    """Test lookup edge cases including partial match and no match"""

    # Test partial match
    tokens = generate_tokens(1000, "cpu")

    # Only add first 3 chunks using token_database to generate keys
    chunk_count = 0
    for start, end, key in kv_controller.token_database.process_tokens(
        tokens, make_key=False
    ):
        if chunk_count >= 3:
            break
        assert isinstance(key, int)
        admit_msg = KVAdmitMsg(
            instance_id="test_instance",
            worker_id=0,
            key=key,
            location="LocalCPUBackend",
        )
        await kv_controller.admit(admit_msg)
        chunk_count += 1

    # Test lookup should only match first 3 chunks
    lookup_msg = LookupMsg(event_id="test_lookup", tokens=tokens.tolist())
    result = await kv_controller.lookup(lookup_msg)

    assert "test_instance" in result.layout_info
    location, end = result.layout_info["test_instance"]
    assert end == 3 * 256

    # Test no match
    empty_tokens = []
    empty_lookup_msg = LookupMsg(event_id="test_lookup", tokens=empty_tokens)
    empty_result = await kv_controller.lookup(empty_lookup_msg)
    assert empty_result.layout_info == {}

    # Test no match with tokens but no KV chunks
    no_match_tokens = generate_tokens(500, "cpu")
    no_match_msg = LookupMsg(event_id="test_lookup", tokens=no_match_tokens.tolist())
    no_match_result = await kv_controller.lookup(no_match_msg)
    assert no_match_result.layout_info == {}


@pytest.mark.asyncio
async def test_full_lookup_edge_cases():
    kv_controller = KVController()
    """Test full lookup edge cases including partial match and no match"""

    # Test partial match
    tokens = generate_tokens(1000, "cpu")

    # Only add first 3 chunks using token_database to generate keys
    chunk_count = 0
    for start, end, key in kv_controller.token_database.process_tokens(
        tokens, make_key=False
    ):
        if chunk_count >= 3:
            break
        assert isinstance(key, int)
        # multiple locations
        admit_msg = KVAdmitMsg(
            instance_id="test_instance",
            worker_id=0,
            key=key,
            location="LocalDiskBackend",
        )
        await kv_controller.admit(admit_msg)
        admit_msg = KVAdmitMsg(
            instance_id="test_instance",
            worker_id=0,
            key=key,
            location="LocalCPUBackend",
        )
        await kv_controller.admit(admit_msg)
        chunk_count += 1

    # Test full lookup should only match first 3 chunks
    full_lookup_msg = FullLookupMsg(event_id="test_full_lookup", tokens=tokens.tolist())
    full_result = await kv_controller.full_lookup(full_lookup_msg)

    assert len(full_result.matched_info) == 1
    instance_id, cache_list = full_result.matched_info[0]
    assert instance_id == "test_instance"
    assert len(cache_list) == 3
    assert full_result.chunk_size == 256
    for cache_chunk in cache_list: # make sure CPU is always prioritized
        assert cache_chunk[0] == "LocalCPUBackend"

    # Test no match
    empty_tokens = []
    empty_full_lookup_msg = FullLookupMsg(
        event_id="test_full_lookup", tokens=empty_tokens
    )
    empty_full_result = await kv_controller.full_lookup(empty_full_lookup_msg)
    assert empty_full_result.matched_info == []

    # Test no match with tokens but no KV chunks
    no_match_tokens = generate_tokens(500, "cpu")
    no_match_full_msg = FullLookupMsg(
        event_id="test_full_lookup", tokens=no_match_tokens.tolist()
    )
    no_match_full_result = await kv_controller.full_lookup(no_match_full_msg)
    assert no_match_full_result.matched_info == []


@pytest.mark.asyncio
async def test_edge_cases_and_errors():
    kv_controller = KVController()
    """Test edge cases and error handling"""

    # Generate tokens and get key from token_database
    tokens = generate_tokens(256, "cpu")  # Use chunk_size tokens

    # Get the first key from token_database
    for start, end, key in kv_controller.token_database.process_tokens(
        tokens, make_key=False
    ):
        assert isinstance(key, int)

        # Test duplicate admit for same key
        admit_msg1 = KVAdmitMsg(
            instance_id="instance1", worker_id=0, key=key, location="LocalCPUBackend"
        )
        admit_msg2 = KVAdmitMsg(
            instance_id="instance2", worker_id=0, key=key, location="LocalDiskBackend"
        )

        await kv_controller.admit(admit_msg1)
        await kv_controller.admit(admit_msg2)

        # Verify both KV chunks exist
        assert str(key) in kv_controller.kv_pool
        assert len(kv_controller.kv_pool[str(key)]) == 2

        # Test evict non-existent key
        non_existent_msg = KVEvictMsg(
            instance_id="nonexistent", worker_id=0, key=-1, location="LocalCPUBackend"
        )
        await kv_controller.evict(non_existent_msg)

        # Test deregister non-existent instance
        await kv_controller.deregister("nonexistent", 0)
        break  # Only test with first key


@pytest.mark.asyncio
async def test_concurrent_operations():
    kv_controller = KVController()
    """Test concurrent operations"""
    # Generate tokens for multiple keys
    tokens = generate_tokens(2560, "cpu")  # 10 chunks * 256 tokens

    # Get keys from token_database
    keys = []
    for start, end, key in kv_controller.token_database.process_tokens(
        tokens, make_key=False
    ):
        assert isinstance(key, int)
        keys.append(key)
        if len(keys) >= 10:  # Limit to 10 keys
            break

    # Create multiple concurrent tasks
    async def add_kv_chunk(i):
        key = keys[i]
        admit_msg = KVAdmitMsg(
            instance_id=f"instance_{i}",
            worker_id=i,
            key=key,
            location="LocalCPUBackend",
        )
        await kv_controller.admit(admit_msg)
        return key

    # Concurrently add 10 KV chunks
    tasks = [add_kv_chunk(i) for i in range(10)]
    added_keys = await asyncio.gather(*tasks)

    # Verify all KV chunks added
    for key in added_keys:
        assert str(key) in kv_controller.kv_pool

    # Concurrently remove KV chunks
    async def remove_kv_chunk(i):
        key = keys[i]
        evict_msg = KVEvictMsg(
            instance_id=f"instance_{i}",
            worker_id=i,
            key=key,
            location="LocalCPUBackend",
        )
        await kv_controller.evict(evict_msg)

    remove_tasks = [remove_kv_chunk(i) for i in range(10)]
    await asyncio.gather(*remove_tasks)

    # Verify all KV chunks removed
    for key in added_keys:
        assert key not in kv_controller.kv_pool
