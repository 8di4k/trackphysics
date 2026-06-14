"""Input schema — the types we own and that adapters target.

The core consumes tracks of *nameless* objects. There is no notion here of what an
object *is*: ``class_id`` is an opaque integer, the skeleton is whatever graph the
caller supplies, and a "segment" is just a contiguous window of frames. Any meaning
(what the object is, what a contact signifies) is attached by a domain layer built on
top — never here. See BRIEF.md §6.

The data model is deliberately frame-incremental friendly: a :class:`TrackSequence` is
an ordered list of per-frame :class:`Detection` records, so a future streaming mode can
append detections without reshaping the structures (BRIEF.md §7, §16).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
"""Alias for a float64 ndarray (used throughout for positions, velocities, bboxes)."""


@dataclass
class Detection:
    """A single object observation in a single frame.

    All pixel quantities use the image coordinate convention: ``x`` rightward,
    ``y`` downward, origin at the top-left corner.
    """

    frame: int
    """Frame index this detection belongs to."""

    bbox: FloatArray
    """Bounding box, shape ``(4,)``, in **pixels**, ``xyxy`` order
    ``(x_min, y_min, x_max, y_max)``."""

    track_id: int
    """Identity assigned by the upstream tracker."""

    score: float | None = None
    """Detector confidence in ``[0, 1]`` if available, else ``None``."""

    keypoints: FloatArray | None = None
    """Optional keypoints, shape ``(K, 2)`` as ``[x, y]`` or ``(K, 3)`` as
    ``[x, y, conf]``, in **pixels**."""

    class_id: int | None = None
    """OPAQUE to the core. A domain layer maps it to meaning; the engine never
    interprets it (BRIEF.md §6, hook 1)."""

    def __post_init__(self) -> None:
        # np.array (not np.asarray) so the Detection OWNS its bbox: asarray returns the same
        # object for an already-float64 input, which would alias the caller's array (and the
        # adapter's shared buffer), letting in-place smoothing/imputation corrupt the input
        # or vice-versa (SCHEMA-ALIASING, BRIEF.md §4 isolation).
        self.bbox = np.array(self.bbox, dtype=np.float64)
        if self.bbox.shape != (4,):
            raise ValueError(f"bbox must have shape (4,), got {self.bbox.shape}")
        if self.keypoints is not None:
            kp = np.array(self.keypoints, dtype=np.float64)  # copy: own it (see bbox note)
            if kp.ndim != 2 or kp.shape[1] not in (2, 3):
                raise ValueError(
                    f"keypoints must have shape (K, 2) or (K, 3), got {kp.shape}"
                )
            self.keypoints = kp

    @property
    def center(self) -> FloatArray:
        """Pixel centroid of the bounding box, shape ``(2,)`` as ``[x, y]``."""
        x0, y0, x1, y1 = self.bbox
        return np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=np.float64)


@dataclass
class SkeletonGraph:
    """Keypoint connectivity for an articulated object.

    There is NO hardcoded skeleton (human or otherwise). The caller supplies the
    graph per object type (BRIEF.md §6, hook 5). Kinematics derived from it (joint
    angles, angular velocities) are scale-invariant and therefore valid at any tier.
    """

    num_keypoints: int
    edges: list[tuple[int, int]]
    """Undirected connectivity as ``(i, j)`` index pairs into the keypoint array."""

    def __post_init__(self) -> None:
        for i, j in self.edges:
            if not (0 <= i < self.num_keypoints and 0 <= j < self.num_keypoints):
                raise ValueError(
                    f"edge ({i}, {j}) out of range for num_keypoints={self.num_keypoints}"
                )


@dataclass
class TrackSequence:
    """One object's track: detections ordered by frame, plus timing.

    PRECONDITION (v0.1): a static camera (BRIEF.md §7). If camera motion is present,
    consumers must drop confidence or compensate upstream — the engine must never
    silently emit wrong metric output for a moving camera.
    """

    detections: list[Detection]
    fps: float
    timestamps: FloatArray | None = None
    """Explicit per-frame timestamps in seconds. Overrides ``fps`` when present."""

    skeleton: SkeletonGraph | None = None
    image_size: tuple[int, int] | None = None
    """``(W, H)`` in pixels; helps normalization for the relative-3D lift."""

    def __post_init__(self) -> None:
        if self.timestamps is not None:
            self.timestamps = np.array(self.timestamps, dtype=np.float64)  # own it (copy)
            if self.timestamps.ndim != 1:
                raise ValueError(
                    f"timestamps must be 1-D, got shape {self.timestamps.shape}"
                )
            if self.timestamps.shape[0] != len(self.detections):
                raise ValueError(
                    f"timestamps length {self.timestamps.shape[0]} must match detections "
                    f"count {len(self.detections)}"
                )
        if len(self.detections) >= 2:
            # The contract is "ordered by STRICTLY INCREASING frame": one object occupies
            # each frame at most once. Enforcing it at the boundary stops out-of-order OR
            # DUPLICATE frames from silently producing degenerate velocity (duplicate frames
            # -> dt=0 -> divide-by-zero inf/nan velocity) and bogus gap flags deep in the
            # pipeline. Adapters sort for you; collapse duplicates upstream if a tracker
            # double-reports an object in one frame (SCHEMA-DUP-FRAMES).
            frames = np.array([d.frame for d in self.detections], dtype=np.int64)
            if bool(np.any(np.diff(frames) <= 0)):
                raise ValueError(
                    "detections must be ordered by strictly increasing frame; got "
                    "out-of-order or duplicate frames (use an adapter such as from_generic, "
                    "which sorts, or sort/de-duplicate first)"
                )

    def __len__(self) -> int:
        return len(self.detections)

    @property
    def frames(self) -> npt.NDArray[np.int64]:
        """Frame indices, shape ``(T,)``."""
        return np.array([d.frame for d in self.detections], dtype=np.int64)

    def times(self) -> FloatArray:
        """Per-detection time in seconds.

        Uses explicit ``timestamps`` when given; otherwise derives time from frame
        indices and ``fps`` (so non-contiguous frames keep correct spacing).
        """
        if self.timestamps is not None:
            return self.timestamps.copy()  # pure value accessor: never hand out the internal ref
        if self.fps <= 0:
            raise ValueError("fps must be positive when timestamps are absent")
        return self.frames.astype(np.float64) / self.fps

    def centers(self) -> FloatArray:
        """Pixel bbox centroids, shape ``(T, 2)``."""
        if not self.detections:
            return np.empty((0, 2), dtype=np.float64)
        return np.stack([d.center for d in self.detections])

    def bboxes(self) -> FloatArray:
        """Per-detection bounding boxes, shape ``(T, 4)`` in ``xyxy`` pixels.

        The apparent-size channel (object-size-as-ruler, the relative-3D depth proxy) reads
        box *extent* from here; ``centers()`` reads only position. Empty track -> ``(0, 4)``.
        """
        if not self.detections:
            return np.empty((0, 4), dtype=np.float64)
        return np.stack([d.bbox for d in self.detections])


@dataclass
class Segment:
    """A contiguous window of a track, classified only by a *generic* motion kind.

    ``kind`` is kinematic, never semantic: e.g. ``"ballistic"``, ``"static"``,
    ``"unknown"``. A domain layer decides what, if anything, the window *means*.
    """

    start_frame: int
    end_frame: int
    kind: str = "unknown"
    indices: npt.NDArray[np.int64] | None = None
    """Indices into the owning track's ``detections`` list, if materialized."""

    source_track: TrackSequence | None = None
    """The track this segment indexes into. A detector attaches it so a stateless
    ``fit(segment, ctx)`` can recover the pixel observations without the preset holding
    per-track instance state (avoids the shared-singleton hazard). Optional/backward-
    compatible; defaults to ``None``."""

    def __post_init__(self) -> None:
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"end_frame ({self.end_frame}) precedes start_frame ({self.start_frame})"
            )
        if self.indices is not None:
            self.indices = np.asarray(self.indices, dtype=np.int64)


__all__ = ["Detection", "FloatArray", "Segment", "SkeletonGraph", "TrackSequence"]
