from typing import Literal, Dict, Iterator
import time
import datetime
import copy
from collections import deque
from threading import RLock
from fsspec import AbstractFileSystem


class TrieNodeValue:
    def __init__(
        self,
        buf: bytearray | None,
        creation_time: float,
    ):
        self.modified = creation_time
        self.created = creation_time
        self.buf = buf

    @classmethod
    def create_dir_node(cls):
        return cls(None, time.time())

    @classmethod
    def create_file_node(cls):
        return cls(bytearray(), time.time())

    def clear(self):
        if self.buf is None:
            return
        else:
            self.buf = bytearray()

    @property
    def size(self) -> int:
        return len(self.buf) if self.buf is not None else 0

    @property
    def is_file(self) -> bool:
        return self.buf is not None

    @property
    def is_dir(self) -> bool:
        return self.buf is None


class TrieNode:
    def __init__(self, name: str, value: TrieNodeValue):
        self.name: str = name
        self.children: Dict[str, "TrieNode"] = {}
        self.value = value


class TrieMemoryFileSystem(AbstractFileSystem):
    protocol = "memory"
    root_marker = "/"

    def __init__(self, *_, **__):
        super().__init__()
        self._root = TrieNode(
            self.root_marker, TrieNodeValue.create_dir_node()
        )
        self._pathmap = {self.root_marker: self._root}
        self._lock = RLock()  # upgrade to RW lock if needed

    def _now(self) -> float:
        return time.time()

    def _write_lock(self) -> RLock:
        return self._lock

    def _split(self, path: str):
        stripped = self._strip_protocol(path)
        assert isinstance(stripped, str)
        path = stripped.strip("/")
        return [] if not path else path.split("/")

    def _norm(self, path: str) -> str:
        stripped = self._strip_protocol(path)
        assert isinstance(stripped, str)
        return stripped.strip("/")

    def _with_root(self, path: str) -> str:
        """Ensure path includes the root marker."""
        if path.startswith(self.root_marker):
            return path
        return f"{self.root_marker}{path}"

    def _record(self, name: str, node: TrieNode):

        return {
            "name": name,
            "size": node.value.size,
            "type": "directory" if node.value.is_dir else "file",
            "created": node.value.created,
            "modified": node.value.modified,
        }

    def _walk_to(
        self, path: str, create: bool = False, as_dir: bool = False
    ) -> TrieNode | None:
        parts = self._split(path)
        node = self._root
        cur = []
        for i, p in enumerate(parts):
            cur_path = "/".join(cur + [p])
            child = node.children.get(p)
            if child is None:
                if not create:
                    return None
                if i == len(parts) - 1:
                    child = TrieNode(
                        name=p,
                        value=(
                            TrieNodeValue.create_dir_node()
                            if as_dir
                            else TrieNodeValue.create_file_node()
                        ),
                    )
                else:
                    child = TrieNode(
                        name=p, value=TrieNodeValue.create_dir_node()
                    )
                node.children[p] = child
                self._pathmap[cur_path] = child
                child.value.modified = self._now()

            node = child
            cur.append(p)

        return node

    def open(
        self,
        path,
        mode: Literal["rb", "wb", "ab", "rb+", "wb+", "xb", "xb+"] = "rb",
        **kwargs,
    ):
        with self._lock:
            path = self._norm(path)
            node = self._pathmap.get(path)
            if "x" in mode:
                if node is not None:
                    raise FileExistsError(path)
                mode = "wb+" if "+" in mode else "wb"
            if node is None:
                node = self._walk_to(
                    path, create="w" in mode or "a" in mode, as_dir=False
                )
            if node is None:
                raise FileNotFoundError(path)

            if node.value.is_dir:
                raise IsADirectoryError(path)

            return InMemFile(self, node, mode)

    def ls(self, path: str, detail: bool = True, **kwargs) -> list[dict]:
        with self._lock:
            path = self._norm(path)
            node = self._pathmap.get(path)
            if node is None:
                raise FileNotFoundError(path)
            if node.value.is_file:
                rec = self._record(path, node)
                return [rec] if detail else [rec["name"]]
            # directory
            out = []
            for name, child in node.children.items():
                full = self._with_root(f"{path}/{name}")
                out.append(self._record(full, child))
            return out if detail else sorted(r["name"] for r in out)

    def info(self, path, **kwargs):
        with self._lock:
            path = self._norm(path)
            node = self._pathmap.get(path)
            if node is None:
                raise FileNotFoundError(path)
            return self._record(self._with_root(path), node)

    def exists(self, path):
        with self._lock:
            return self._norm(path) in self._pathmap

    def mkdir(self, path, create_parents=True, **kwargs):
        with self._lock:
            parent = "/".join(self._split(path)[:-1])
            if not create_parents and parent and not self.exists(parent):
                raise FileNotFoundError(parent)
            self._walk_to(path, create=True, as_dir=True)

    def makedirs(self, path, exist_ok=False):
        with self._lock:
            path = self._norm(path)
            existing = self._pathmap.get(path)

            if existing is not None:
                if not exist_ok or existing.value.is_file:
                    raise FileExistsError()
            else:
                self.mkdir(path, create_parents=True)

    def rm_file(self, path, recursive=False, **kwargs):
        with self._lock:
            path = self._norm(path)
            parts = self._split(path)
            if not parts:
                raise PermissionError("cannot remove root")
            # find parent
            parent_node = self._root
            for p in parts[:-1]:
                parent_node = parent_node.children.get(p)
                if parent_node is None:
                    raise FileNotFoundError(path)
            name = parts[-1]
            target = parent_node.children.get(name)
            if target is None:
                raise FileNotFoundError(path)
            if target.value.is_dir and target.children and not recursive:
                raise OSError("directory not empty")
            # delete subtree
            stack = deque([(path, target)])
            while stack:
                pth, nd = stack.pop()
                self._pathmap.pop(pth, None)
                if nd.children:
                    for c_name, c_node in list(nd.children.items()):
                        stack.append((f"{pth}/{c_name}", c_node))
                nd.children.clear()
            parent_node.children.pop(name)

    # def mv(self, path1: str, path2: str, **kwargs) -> None:
    #    with self._lock:
    #        path1 = self._norm(path1)
    #        path2 = self._norm(path2)
    #
    #        src_node = self._pathmap.get(path1)
    #        if src_node is None:
    #            raise FileNotFoundError(path1)
    #
    #        dst_parent_path = "/".join(self._split(path2)[:-1])
    #        if dst_parent_path:
    #            dst_parent_node = self._pathmap.get(dst_parent_path)
    #            if not dst_parent_node:
    #                dst_parent_node = self._walk_to(
    #                    dst_parent_path, create=True, as_dir=True
    #                )
    #        else:
    #            dst_parent_node = self._root
    #        assert dst_parent_node is not None
    #
    #        # If destination exists, remove it
    #        if self.exists(path2):
    #            self.rm(path2, recursive=True)
    #
    #        # Detach from old parent
    #        src_parent_path = "/".join(self._split(path1)[:-1])
    #        src_parent = (
    #            self._pathmap.get(src_parent_path)
    #            if src_parent_path
    #            else self._root
    #        )
    #        assert src_parent is not None
    #        src_parent.children.pop(self._split(path1)[-1])
    #
    #        # Attach to new parent
    #        dst_parent_node.children[self._split(path2)[-1]] = src_node
    #
    #        # Reindex
    #        self._reindex_subtree(path1, path2, src_node)

    def walk(
        self, path: str, topdown: bool = True, maxdepth: int | None = None
    ) -> Iterator[tuple[str, list[str], list[str]]]:
        """Generate (dirpath, dirnames, filenames) tuples."""
        with self._lock:
            path = self._norm(path)
            start_node = self._pathmap.get(path)
            if not start_node:
                raise FileNotFoundError(path)

            def _walk(node: TrieNode, cur_path: str, depth: int):
                dirs: list[str] = []
                files: list[str] = []
                for name, child in node.children.items():
                    if child.value.is_dir:
                        dirs.append(name)
                    else:
                        files.append(name)

                if topdown:
                    yield self._with_root(cur_path), dirs, files

                if maxdepth is None or depth < maxdepth:
                    for d in dirs:
                        yield from _walk(
                            node.children[d],
                            f"{cur_path}/{d}" if cur_path else d,
                            depth + 1,
                        )

                if not topdown:
                    yield self._with_root(cur_path), dirs, files

            yield from _walk(start_node, self._norm(path), 1)

    def find(
        self,
        path: str,
        withdirs: bool = False,
        detail: bool = False,
        maxdepth: int | None = None,
    ) -> list | dict:
        """List all entries under path recursively."""
        with self._lock:
            out_names = []
            out_details = {}
            for dirpath, dirnames, filenames in self.walk(
                path, maxdepth=maxdepth
            ):
                if withdirs:
                    for d in dirnames:
                        full = f"{dirpath}/{d}" if dirpath else d
                        if detail:
                            out_details[full] = self._record(
                                full, self._pathmap[self._norm(full)]
                            )
                        else:
                            out_names.append(full)

                for f in filenames:
                    full = f"{dirpath}/{f}" if dirpath else f
                    if detail:
                        out_details[full] = self._record(
                            full, self._pathmap[self._norm(full)]
                        )
                    else:
                        out_names.append(full)
            if detail:
                return out_details
            else:
                return out_names

    def _reindex_subtree(self, old: str, new: str, node: TrieNode) -> None:
        # walk subtree and rewrite keys in _pathmap
        stack = deque([("", node)])
        self._pathmap.pop(old, None)
        self._pathmap[new] = node
        while stack:
            rel, nd = stack.pop()
            base = f"{new}/{rel}" if rel else new
            old_base = f"{old}/{rel}" if rel else old
            for k, ch in nd.children.items():
                p = f"{base}/{k}" if base else k
                old_p = f"{old_base}/{k}" if old_base else k
                self._pathmap[p] = ch
                self._pathmap.pop(old_p, None)
                stack.append((f"{rel}/{k}" if rel else k, ch))

    def cp_file(self, path1, path2, **kwargs):
        """Copy a file's contents and metadata to a new path."""
        with self._lock:
            path1 = self._norm(path1)
            path2 = self._norm(path2)

            # Ensure source exists and is a file
            src_node = self._pathmap.get(path1)
            if src_node is None:
                raise FileNotFoundError(path1)
            if src_node.value.is_dir:
                return
                # raise IsADirectoryError(path1)

            # Create destination parent if needed
            dst_parent_path = "/".join(self._split(path2)[:-1])
            if dst_parent_path and not self.exists(dst_parent_path):
                self._walk_to(dst_parent_path, create=True, as_dir=True)

            # If destination exists and is a dir, error
            dst_node = self._pathmap.get(path2)
            if dst_node and dst_node.value.is_dir:
                raise IsADirectoryError(path2)

            # Create or overwrite destination file node
            new_value = TrieNodeValue(
                buf=copy.deepcopy(src_node.value.buf),
                creation_time=self._now(),
            )
            new_value.modified = src_node.value.modified  # Preserve mod time
            dst_node = TrieNode(self._split(path2)[-1], new_value)

            # Attach to parent
            parent_node = (
                self._pathmap.get(dst_parent_path)
                if dst_parent_path
                else self._root
            )
            assert parent_node is not None
            parent_node.children[self._split(path2)[-1]] = dst_node

            # Update pathmap
            self._pathmap[path2] = dst_node

    def created(self, path):
        """Return the created timestamp of a file or directory as a
        datetime.datetime (UTC)."""
        with self._lock:
            path = self._norm(path)
            node = self._pathmap.get(path)
            if node is None:
                raise FileNotFoundError(path)
            return datetime.datetime.fromtimestamp(
                node.value.created, tz=datetime.timezone.utc
            )

    def modified(self, path):
        """Return the modified timestamp of a file or directory as a
        datetime.datetime (UTC)."""
        with self._lock:
            path = self._norm(path)
            node = self._pathmap.get(path)
            if node is None:
                raise FileNotFoundError(path)
            return datetime.datetime.fromtimestamp(
                node.value.modified, tz=datetime.timezone.utc
            )


class InMemFile:
    def __init__(
        self,
        fs: TrieMemoryFileSystem,
        node: TrieNode,
        mode: Literal["rb", "wb", "ab", "rb+", "wb+", "xb", "xb+"],
    ):
        self.fs = fs
        self.node = node
        self.mode = mode
        self.pos = 0
        if "w" in mode:
            self.buf = bytearray()
        else:
            if node.value is not None and node.value.buf is not None:
                self.buf = node.value.buf
            else:
                self.buf = bytearray()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def read(self, n=-1):
        if n is None or n < 0:
            n = len(self.buf) - self.pos
        out = bytes(self.buf[slice(self.pos, self.pos + n)])
        self.pos += len(out)
        return out

    def write(self, b):
        end = self.pos + len(b)
        if end > len(self.buf):
            self.buf.extend(b"\x00" * (end - len(self.buf)))
        self.buf[slice(self.pos, end)] = b
        self.pos = end
        return len(b)

    def seek(self, offset, whence=0):
        if whence == 1:
            offset += self.pos
        elif whence == 2:
            offset += len(self.buf)
        if offset < 0:
            raise ValueError("negative seek")
        self.pos = offset
        return self.pos

    def close(self):
        if "w" in self.mode or "+" in self.mode or "a" in self.mode:
            with self.fs._write_lock():
                assert self.node.value is not None
                self.node.value.buf = self.buf

                self.node.value.modified = self.fs._now()
