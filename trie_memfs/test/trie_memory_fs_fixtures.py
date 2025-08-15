import pytest

from trie_memfs.trie_memory_fs import TrieMemoryFileSystem
from fsspec.tests.abstract import AbstractFixtures


class MemoryFixtures(AbstractFixtures):
    @pytest.fixture(scope="class")
    def fs(self):
        m = TrieMemoryFileSystem("memory")
        yield m

    @pytest.fixture
    def fs_join(self):
        return lambda *args: "/".join(args)

    @pytest.fixture
    def fs_path(self):
        return ""
