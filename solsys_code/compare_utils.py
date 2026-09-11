import hashlib

import pandas as pd
from pandas import util
from pandas.testing import assert_frame_equal


def _df_hash(df: pd.DataFrame) -> str:
    """Return an md5 hash for a DataFrame that is stable across row/column order.

    The DataFrame is sorted by index and columns before hashing to make
    comparisons deterministic.
    """
    df2 = df.copy()
    df2 = df2.sort_index(axis=0).sort_index(axis=1)
    arr = util.hash_pandas_object(df2, index=True).values
    return hashlib.md5(arr.tobytes()).hexdigest()


def compare_ades_with_csv(
    ades_df: pd.DataFrame,
    csv_path: str,
    *,
    index_col=None,
    align_columns: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_dtype: bool = False,
    return_diff_rows: int = 20,
    float_decimals=None,
) -> dict:
    """Compare a DataFrame (`ades_df`) with a CSV on disk.

    Returns a dictionary with keys:
    - `match` (bool): True if frames match under given tolerances.
    - `message` (str): assertion message or success message.
    - `hash_ades`, `hash_disk`: md5 hashes of each frame (post-sort).
    - `diff` or `diff_rows`: a small sample of differences when present.

    Notes:
    - Column order and index are aligned to `ades_df` by default.
    - Use `rtol`/`atol` to control floating-point tolerance.
    """
    disk_df = pd.read_csv(csv_path, index_col=index_col)

    # Work on copies so we don't mutate caller data
    left = ades_df.copy()
    right = disk_df.copy()

    if align_columns:
        # Reindex columns to match `ades_df` order (missing columns become NaN)
        right = right.reindex(columns=left.columns)

    # Align index to ades_df (missing rows become NaN)
    right = right.reindex(index=left.index)

    # Optionally round float columns to a given number of decimals before comparing.
    # `float_decimals` may be an int (apply to all float columns) or a dict
    # mapping column name -> decimals.
    if float_decimals is not None:
        if isinstance(float_decimals, int):
            # find float columns present in both frames
            float_cols = left.select_dtypes(include=['float']).columns
            for c in float_cols:
                if c in right.columns:
                    left[c] = left[c].round(float_decimals)
                    right[c] = right[c].round(float_decimals)
        elif isinstance(float_decimals, dict):
            for c, dec in float_decimals.items():
                if c in left.columns:
                    left[c] = left[c].round(dec)
                if c in right.columns:
                    right[c] = right[c].round(dec)
        else:
            raise ValueError('float_decimals must be int, dict, or None')

    result = {'match': False}
    result['hash_ades'] = _df_hash(left)
    result['hash_disk'] = _df_hash(right)

    try:
        assert_frame_equal(left, right, check_dtype=check_dtype, rtol=rtol, atol=atol)
    except AssertionError as e:
        result['message'] = str(e)
        # Try to produce a compact per-cell diff
        try:
            diff = left.compare(right, keep_shape=False, keep_equal=False)
            result['diff'] = diff.head(return_diff_rows)
        except Exception:
            # Fallback: rows that contain any difference
            mask = (left != right) & ~(left.isna() & right.isna())
            rows = left[mask.any(axis=1)]
            result['diff_rows'] = rows.head(return_diff_rows)
        return result

    result['match'] = True
    result['message'] = 'DataFrames match'
    return result


__all__ = ['compare_ades_with_csv', '_df_hash']
