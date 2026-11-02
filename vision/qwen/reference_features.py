"""Reuse frozen reference-image features; targets are always encoded afresh.

The pinned Qwen3-VL vision encoder has independent attention for each image.
Cache entries are bound to source pixels, processor grid and tensor dtype.
"""
from __future__ import annotations


class ReferenceFeatureCache:
    def __init__(self, original, concatenate, max_entries=48):
        self.original = original
        self.concatenate = concatenate
        self.max_entries = max_entries
        self.image_keys = []
        self.entries = {}

    def __call__(self, pixel_values, image_grid_thw, **kwargs):
        grids = image_grid_thw.tolist()
        if len(grids) != len(self.image_keys):
            return self.original(pixel_values, image_grid_thw, **kwargs)
        offset, outputs = 0, []
        for index, (grid, source) in enumerate(zip(grids, self.image_keys)):
            count = grid[0] * grid[1] * grid[2]
            key = (source, tuple(grid), str(pixel_values.dtype)) if source else None
            output = self.entries.get(key) if key else None
            if output is None:
                output = self.original(pixel_values[offset:offset + count], image_grid_thw[index:index + 1],
                                       **{**kwargs, 'return_dict': True})
                if key:
                    if len(self.entries) >= self.max_entries:
                        self.entries.pop(next(iter(self.entries)))
                    self.entries[key] = output
            outputs.append(output)
            offset += count
        if not outputs:
            return self.original(pixel_values, image_grid_thw, **kwargs)
        prototype = outputs[-1]
        return type(prototype)(pooler_output=tuple(output.pooler_output[0] for output in outputs),
                               deepstack_features=[self.concatenate([output.deepstack_features[layer]
                                                                     for output in outputs], dim=0)
                                                   for layer in range(len(prototype.deepstack_features))])
