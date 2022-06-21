"""
Copied from
https://github.com/FNNDSC/chris_plugin/blob/bb4bf231221e7777df4d0836774a8f4c3f0afc8a/chris_plugin/mapper.py

chris_plugin does not support Python 3.6, but just mapper.py should work.
"""
import sys
from pathlib import Path
from typing import Callable, Iterable, Iterator, Tuple, Optional
from dataclasses import dataclass
import logging

_logger = logging.getLogger(__name__)
NameMapper = Callable[[Path, Path], Path]

# Private Helpers
########


def _verbatim(input_file: Path, output_dir: Path) -> Path:
    return output_dir / input_file


def _include_all(_) -> bool:
    return True


def _curry_suffix(suffix: str) -> NameMapper:
    def append_suffix(input_file: Path, output_dir: Path) -> Path:
        return (output_dir / input_file).with_suffix(suffix)

    return append_suffix


# Public Classes
########


@dataclass(frozen=True)
class PathMapper(Iterable[Tuple[Path, Path]]):
    """
    An iterator which discovers input files in a directory and maps
    them to output path names.

    For example, an input file `/share/incoming/a/b/c.txt` will be
    mapped to `/share/outgoing/a/b/c.txt`.

    A common use case would be *ChRIS* *ds* plugins which operate
    on individual input files.

    Examples
    --------

    Examples in this section are advanced tips and tricks. For
    common use cases, see `PathMapper.file_mapper`.

    Use [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor)
    to parallelize subprocesses (akin to the usage of GNU
    [parallel](https://www.gnu.org/software/parallel/)):

    ```python
    import subprocess as sp
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        for input_file, output_path in PathMapper(input_dir, output_dir):
            pool.submit(sp.run, ['external_command', input_file, output_path])
    ```

    Hint: `len(os.sched_getaffinity(0))` gets the number of CPUs available
    to a containerized process (which can be limited by, for instance,
    `docker run --cpuset-cpus 0-3`)

    Add a progress bar with [tqdm](https://github.com/tqdm/tqdm):

    ```python
    from tqdm import tqdm

    with tqdm(PathMapper(input_dir, output_dir)) as bar:
        for input_file, output_path in bar:
            do_something(input_file, output_path)
    ```
    """

    input_dir: Path
    """Directory containing input files"""
    output_dir: Path
    """Directory for output files to be written to"""

    name_mapper: NameMapper = _verbatim
    """
    Specify a custom function which produces an output file name given the
    input path relative to `input_dir`, and `output_dir`.
    """

    glob: str = "**/*"
    """
    File name pattern matching input files in `input_dir`.
    """

    parents: bool = True
    """
    If `True`, create parent directories of output paths as needed.
    """

    fail_if_empty: bool = True
    """
    Exit the program if no input files are found.
    """

    filter: Callable[[Path], bool] = _include_all
    """
    Decides whether a given subpath of input directory should be in the input space.
    """

    def __post_init__(self):
        if not self.input_dir.is_dir():
            raise ValueError()
        if not self.output_dir.is_dir() and self.output_dir.exists():
            raise ValueError()

    @classmethod
    def file_mapper(
        cls,
        input_dir: Path,
        output_dir: Path,
        glob: str = "**/*",
        name_mapper: NameMapper = _verbatim,
        suffix: Optional[str] = None,
        fail_if_empty: bool = True,
        filter: Callable[[Path], bool] = _include_all,
    ) -> "PathMapper":
        """
        Constructor for `PathMapper` for working with files.

        Examples
        --------

        Copy all files from `input_dir` to `output_dir`:

        ```python
        for input_file, output_file in PathMapper.file_mapper(input_dir, output_dir):
            shutil.copy(input_file, output_file)
        ```

        Avoid clobbering (overwriting existing files):

        ```python
        for input_file, output_file in PathMapper.file_mapper(input_dir, output_dir):
            if output_file.exists():
                print(f'error, file exists: {output_file}')
                sys.exit(1)
            shutil.copy(input_file, output_file)
        ```

        Call the function `segmentation` on only NIFTI files, and rename output
        file names to end with `.seg.nii`:

        ```python
        mapper = PathMapper.file_mapper(input_dir, output_dir, glob='**/*.nii', suffix='.seg.nii')
        for input_file, output_file in mapper:
            segmentation(input_file, output_file)
        ```

        Parameters
        ----------

        suffix: str
            Syntactical sugar for `name_mapper`. If specified, a `name_mapper` is created
            which replaces the file extension of input files with the given value for `suffix`.

        See field documentation for other arguments.
        """
        if suffix is not None:
            if name_mapper is not _verbatim:
                raise ValueError('Only one of ["suffix", "name_mapper"] can be given')
            name_mapper = _curry_suffix(suffix)
        return cls(
            input_dir=input_dir,
            output_dir=output_dir,
            glob=glob,
            name_mapper=name_mapper,
            fail_if_empty=fail_if_empty,
            filter=lambda p: p.is_file() and filter(p),
        )

    @classmethod
    def dir_mapper_shallow(
        cls,
        input_dir: Path,
        output_dir: Path,
        name_mapper: NameMapper = _verbatim,
        fail_if_empty: bool = True,
        filter: Callable[[Path], bool] = _include_all,
    ) -> "PathMapper":
        """
        Constructor for `PathMapper` for working with immediate subdirectories of `input_dir`.

        For instance, if the `input_dir` contains the subdirectories `a/`, `b/c/`, and the files
        `a/d.txt`, `e.txt`, the `PathMapper` will visit `a/` and `b/` (but not `b/c/` nor `e.txt`).

        Parameters
        ----------

        See field documentation.
        """
        return cls(
            glob="*/",  # doesn't seem like trailing slash is helpful here
            input_dir=input_dir,
            output_dir=output_dir,
            name_mapper=name_mapper,
            fail_if_empty=fail_if_empty,
            filter=lambda p: p.is_dir() and filter(p),
        )

    @classmethod
    def dir_mapper_deep(
        cls,
        input_dir: Path,
        output_dir: Path,
        name_mapper: NameMapper = _verbatim,
        fail_if_empty: bool = True,
        filter: Callable[[Path], bool] = _include_all,
    ) -> "PathMapper":
        """
        Constructor for `PathMapper` for working with subpaths of `input_dir` which are
        directories that do not further contain subdirectories.

        For instance, if the `input_dir` contains the subdirectories `a/`, `b/c/`, and the files
        `a/d.txt`, `e.txt`, the `PathMapper` will visit `a/` and `c/` (but not `b/` nor `e.txt`).

        Parameters
        ----------

        See field documentation.
        """
        return cls(
            glob="**/",
            input_dir=input_dir,
            output_dir=output_dir,
            name_mapper=name_mapper,
            fail_if_empty=fail_if_empty,
            filter=lambda p: cls._is_deep_dir(p) and filter(p),
        )

    @staticmethod
    def _is_deep_dir(p: Path) -> bool:
        """
        :return: True if given path is a directory which does not contain subdirectories
        """
        if not p.is_dir():
            return False
        dirs = filter(lambda subpath: subpath.is_dir(), p.glob("*/"))
        return next(dirs, None) is None

    def iter_input(self) -> Iterator[Path]:
        """
        :return: an iterator over input files
        """
        return filter(self.filter, self.input_dir.glob(self.glob))

    def __len__(self):
        return self.count()

    def count(self) -> int:
        """
        Count the number of input paths under `input_dir`.
        """
        return sum(map(lambda _: 1, self.iter_input()))

    def is_empty(self) -> bool:
        return next(self.iter_input(), None) is None

    def __iter__(self) -> Iterator[Tuple[Path, Path]]:
        if self.fail_if_empty and self.is_empty():
            _logger.warning(f'no input found for "{self.input_dir / self.glob}"')
            sys.exit(1)
        for input_path in self.iter_input():
            output_path = self.output_for(input_path)
            if self.parents:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            yield input_path, output_path

    def output_for(self, input_path: Path) -> Path:
        """
        Produce a path under `output_dir` which corresponds to the given `input_path`.

        Parameters
        ----------

        input_path: Path
            A subpath of `self.input_dir`
        """
        rel = input_path.relative_to(self.input_dir)
        return self.name_mapper(rel, self.output_dir)
