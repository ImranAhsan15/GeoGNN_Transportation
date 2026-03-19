"""Optional: make additional incomplete examples from the prepared full road FC."""
import os
import random
import arcpy
from graph_utils import msg

WORK_GDB = r"C:\Users\Imran\Documents\ArcGIS\Projects\MyProject.gdb"
PREP_FULL = os.path.join(WORK_GDB, 'prep_full_road')
OUT_INCOMPLETE = os.path.join(WORK_GDB, 'synthetic_incomplete_road')
REMOVE_RATIO = 0.08
RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)
    arcpy.env.overwriteOutput = True
    if arcpy.Exists(OUT_INCOMPLETE):
        arcpy.management.Delete(OUT_INCOMPLETE)
    arcpy.management.CopyFeatures(PREP_FULL, OUT_INCOMPLETE)
    oids = [r[0] for r in arcpy.da.SearchCursor(OUT_INCOMPLETE, ['OID@'])]
    k = max(1, int(len(oids) * REMOVE_RATIO))
    sel = set(random.sample(oids, k))
    layer = 'tmp_incomplete_lyr'
    arcpy.management.MakeFeatureLayer(OUT_INCOMPLETE, layer)
    oid_field = arcpy.Describe(OUT_INCOMPLETE).OIDFieldName
    where = f"{arcpy.AddFieldDelimiters(OUT_INCOMPLETE, oid_field)} IN ({','.join(map(str, sorted(sel)))})"
    arcpy.management.SelectLayerByAttribute(layer, 'NEW_SELECTION', where)
    arcpy.management.DeleteFeatures(layer)
    msg(f'Removed {len(sel)} segments -> {OUT_INCOMPLETE}')


if __name__ == '__main__':
    main()
