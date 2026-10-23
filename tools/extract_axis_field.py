'''
Extract the on-axis magnetic field from a 3D field map (parquet with columns x, y, z [mm] and Bx, By, Bz [T])
and write it in the coordinates of the electrode stack (TTrap, z = 0 at the start of C0), as used by --B_map.

The tracking engine uses the axial field on the axis and the paraxial radial field B_r = -(r/2) dBz/dz.
The script checks this approximation against the map at the electrode radius.
The field of the AEgIS magnets points from the 5 T magnet towards ELENA (-z, against the flight of the ions to the MCP),
opposite to the direction in the map, so by default the field is reversed (--no-reverse_field keeps the map direction).

usage: python tools/extract_axis_field.py <field_map.parquet> [--stack_offset_mm 1095] [--output modules/field_maps/aegis_axis_field.csv]
'''
import argparse
import os

import numpy as np
import polars as pl

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Extract the on-axis magnetic field from a 3D field map")
    parser.add_argument("field_map", help="parquet file with columns x, y, z [mm], Bx, By, Bz [T]")
    parser.add_argument("--stack_offset_mm", type=float, default=1095.0,
                        help="z of the map minus z of the electrode stack (z_stack = z_map + offset). "
                             "Default: the start of C0 is at z_map = -1095 mm")
    parser.add_argument("--check_radius_mm", type=float, default=16.0, help="radius used to check the paraxial approximation")
    parser.add_argument("--reverse_field", action=argparse.BooleanOptionalAction, default=True,
                        help="reverse the direction of the field (the AEgIS field points from the 5 T magnet towards ELENA, -z)")
    parser.add_argument("--output", default=os.path.join(REPO_DIR, "modules", "field_maps", "aegis_axis_field.csv"))
    args = parser.parse_args()

    r = args.check_radius_mm
    # read only the part of the map close to the axis (the file is large)
    near_axis = (pl.scan_parquet(args.field_map)
                 .filter(pl.col("x").abs() <= r, pl.col("y").abs() <= r)
                 .collect())
    axis = near_axis.filter(pl.col("x") == 0, pl.col("y") == 0).sort("z")
    z = axis["z"].to_numpy()
    Bz = axis["Bz"].to_numpy()
    if len(z) == 0:
        raise ValueError("The map has no points on the axis x = y = 0")

    # check of the paraxial approximation at radius r (y = 0 plane)
    off_axis = near_axis.filter(pl.col("x") == near_axis["x"].filter(near_axis["x"] <= r).max(), pl.col("y") == 0).sort("z")
    r_check = off_axis["x"][0]
    Br_paraxial = -0.5 * r_check * 1e-3 * np.gradient(Bz, z * 1e-3)
    Br_map = off_axis["Bx"].to_numpy()
    Bz_map = off_axis["Bz"].to_numpy()
    print(f"axis: {len(z)} points, z from {z.min():.1f} to {z.max():.1f} mm, Bz max {Bz.max():.4f} T at z = {z[np.argmax(Bz)]:.1f} mm")
    print(f"at r = {r_check:.0f} mm: max |Bz(r) - Bz(0)| = {np.max(np.abs(Bz_map - Bz)):.4f} T, "
          f"max |Br - Br_paraxial| = {np.max(np.abs(Br_map - Br_paraxial)):.4f} T (max |Br| = {np.max(np.abs(Br_map)):.4f} T)")

    # the guiding centre follows the field lines, so the axial field must keep its sign
    # (points with zero or opposite field are artifacts at the edges of the map)
    keep = np.sign(Bz) == np.sign(Bz[np.argmax(np.abs(Bz))])
    if not keep.all():
        print(f"dropped {np.sum(~keep)} points with zero or opposite field at z = {z[~keep]} mm")
    sign = -1.0 if args.reverse_field else 1.0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    (pl.DataFrame({"z_stack_mm": z[keep] + args.stack_offset_mm, "Bz_T": sign * Bz[keep]})
     .write_csv(args.output, float_precision=6))
    print(f"saved {args.output} (z_stack = z_map + {args.stack_offset_mm} mm, field {'reversed' if args.reverse_field else 'as in the map'})")


if __name__ == "__main__":
    main()
