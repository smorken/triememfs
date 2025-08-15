import fsspec.tests.abstract as abstract
from trie_memfs.test.trie_memory_fs_fixtures import MemoryFixtures


class TestMemoryCopy(abstract.AbstractCopyTests, MemoryFixtures):
    pass


class TestMemoryGet(abstract.AbstractGetTests, MemoryFixtures):
    pass


class TestMemoryPut(abstract.AbstractPutTests, MemoryFixtures):
    pass


class TestMemoryPipe(abstract.AbstractPipeTests, MemoryFixtures):
    pass


class TestMemoryOpen(abstract.AbstractOpenTests, MemoryFixtures):
    pass
