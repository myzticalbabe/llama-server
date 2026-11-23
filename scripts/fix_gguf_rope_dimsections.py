"""
Direct binary GGUF patch — fix qwen35.rope.dimension_sections [11,11,10] -> [11,11,10,0].
No GGUF library needed. Pure struct + bytearray manipulation.
"""
import struct, os, sys

SRC = os.environ.get('GGUF_SRC') or r'C:\Users\clint\.lmstudio\models\JeffGreen311\Eve-V2-Unleashed-Qwen3.5-8B-Liberated-4K-4B-Merged\Eve-V2-Unleashed-Qwen3.5-8B-Liberated-4K-4B-Merged-Q4_K_M.gguf'
DST = SRC.replace('.gguf', '-fixed.gguf')

if os.path.exists(DST):
    print(f'{DST} exists. Delete it first to re-patch.')
    sys.exit(0)

with open(SRC, 'rb') as f:
    raw = bytearray(f.read())

# ── Parse GGUF header ──
off = 4  # skip magic
ver, = struct.unpack_from('<I', raw, off); off += 4
tensor_count, = struct.unpack_from('<Q', raw, off); off += 8
kv_count, = struct.unpack_from('<Q', raw, off); off += 8

print(f'GGUF v{ver}, {tensor_count} tensors, {kv_count} KV pairs')

# ── Find the target KV pair ──
TARGET_KEY = b'qwen35.rope.dimension_sections'

# Simple approach: search for the key string in the raw data
pos = raw.find(b'\x1e\x00\x00\x00\x00\x00\x00\x00' + TARGET_KEY)
# Actually, the key is preceded by its length (uint64).
# key_length for our target = 30 = 0x1E
# So we search for: <30 as uint64> + <key bytes>

# More robust: search for just the key
pos = raw.find(TARGET_KEY)
if pos < 0:
    print('ERROR: target key not found')
    sys.exit(1)

print(f'Key "{TARGET_KEY.decode()}" found at byte {pos}')

# The GGUF KV layout before the key:
#   key_length (8 bytes) | key_data (N bytes) | value_type (4 bytes)
# After the key:
#   value_content ...

# Find key_length: it's at pos - 8
key_len, = struct.unpack_from('<Q', raw, pos - 8)
print(f'  key_length = {key_len} (at {pos - 8})')
assert key_len == 30, f'Expected key length 30, got {key_len}'

# After key data: value_type
vt_pos = pos + key_len
val_type, = struct.unpack_from('<I', raw, vt_pos)
print(f'  value_type = {val_type} (at {vt_pos})')
assert val_type == 9, f'Expected array type 9, got {val_type}'

# Array contents: array_type (4) + array_len (8) + elements
arr_type_pos = vt_pos + 4
arr_type, = struct.unpack_from('<I', raw, arr_type_pos)
print(f'  array_type = {arr_type} (at {arr_type_pos})')

arr_len_pos = arr_type_pos + 4
arr_len, = struct.unpack_from('<Q', raw, arr_len_pos)
print(f'  array_len = {arr_len} (at {arr_len_pos})')
assert arr_len == 3, f'Expected array length 3, got {arr_len}'

arr_data_pos = arr_len_pos + 8
elements = struct.unpack_from(f'<{arr_len}I', raw, arr_data_pos)
print(f'  elements = {list(elements)}')

# Position after the array data
old_arr_end = arr_data_pos + arr_len * 4  # 947 + 12 = 959

# ── Patch ──
# 1. Change array length from 3 to 4
struct.pack_into('<Q', raw, arr_len_pos, 4)
# 2. Insert new element (uint32 0) after existing elements
NEW_DATA = struct.pack('<I', 0)
raw[old_arr_end:old_arr_end] = NEW_DATA

print(f'\nPatched sections: [11, 11, 10, 0]')
print(f'Inserted 4 bytes at position {old_arr_end}')

# ── Find and fix all absolute tensor data offsets ──
# The tensor info section starts after all KV pairs.
# Walk through all KV pairs starting from the header end.

# Actually, we know the tensor info starts at the position of the FIRST tensor info.
# Let's search for it by scanning from a reasonable position.

# After the sections KV, the remaining KVs continue. The last KV is followed by
# the tensor info section. The tensor info section contains entries with
# name_length, name, n_dims, dims, and offset.

# Instead of parsing all remaining KVs (which is error-prone due to large arrays),
# let's find where tensor data starts and work backward.

# First, let's find what's at old_arr_end + 4 in the PATCHED file.
# The tensor info structure is: name_len(uint64) + name + n_dims(uint32) + dims[n_dims](uint64) + offset(uint64)

# Search for the last occurrence of the last tensor name pattern
# to find the tensor data boundary.

# Better approach: read the FIRST tensor info entry to validate our position
CHECK_POS = old_arr_end + 4  # after our insertion, where next KV would be
print(f'\nChecking data structure at byte {CHECK_POS}...')

# At CHECK_POS in the patched data, we should have the next KV pair
# Let's parse a few bytes to see what's there
# KV structure: key_len(uint64) + key_data + val_type(uint32) + val
next_kl = struct.unpack_from('<Q', raw, CHECK_POS)[0]
print(f'  next key length = {next_kl} (reasonable: {5 < next_kl < 100})')

# Let's use a different strategy entirely.
# Instead of trying to find tensor info boundaries,
# read ALL offsets from the original file, adjust, and write to patched.

print('\nFinding all tensor offsets in original file...')

# Simple approach: read the original file bytes at known tensor info positions
with open(SRC, 'rb') as f:
    orig = f.read()

# The tensor info starts after the metadata section.
# Parse ALL KVs in the original file to find tensor info boundary.
# We use a simple allocator that tracks the position linearly.

orig_off = 4  # skip magic
ver2, = struct.unpack_from('<I', orig, orig_off); orig_off += 4
tc2, = struct.unpack_from('<Q', orig, orig_off); orig_off += 8
kc2, = struct.unpack_from('<Q', orig, orig_off); orig_off += 8

print(f'  Re-parsing header: {tc2} tensors, {kc2} KVs')

# Quick skip: just advance past all KVs by brute force
# For each KV: key_len(8) + key + val_type(4) + val_content
for i in range(kc2):
    kl, = struct.unpack_from('<Q', orig, orig_off); orig_off += 8
    orig_off += kl  # skip key
    vt, = struct.unpack_from('<I', orig, orig_off); orig_off += 4
    if vt in (0, 1, 7):      # U8, I8, BOOL
        orig_off += 1
    elif vt in (2, 3, 4, 5, 6):  # U16, I16, U32, I32, F32
        orig_off += 4
    elif vt in (10, 11, 12):  # U64, I64, F64
        orig_off += 8
    elif vt == 8:             # STRING
        sl, = struct.unpack_from('<Q', orig, orig_off); orig_off += 8
        orig_off += sl
    elif vt == 9:             # ARRAY
        at, = struct.unpack_from('<I', orig, orig_off); orig_off += 4
        al, = struct.unpack_from('<Q', orig, orig_off); orig_off += 8
        if at in (0, 1, 7):      # array of U8/I8/BOOL
            orig_off += al * 1
        elif at in (2, 3):       # array of U16/I16
            orig_off += al * 2
        elif at in (4, 5, 6):    # array of U32/I32/F32
            orig_off += al * 4
        elif at in (10, 11, 12): # array of U64/I64/F64
            orig_off += al * 8
        elif at == 8:            # array of strings
            for _ in range(al):
                sl, = struct.unpack_from('<Q', orig, orig_off); orig_off += 8
                orig_off += sl
        else:
            print(f'  Unknown array element type {at} at KV[{i}]')
            sys.exit(1)
    else:
        print(f'  Unknown value type {vt} at KV[{i}]')
        sys.exit(1)

# orig_off now points to tensor info section in ORIGINAL file
print(f'Tensor info at original byte {orig_off}')
print(f'Remaining bytes in file: {len(orig) - orig_off}')

# ── Now we know the tensor info boundary ──
# In the patched file, everything from byte old_arr_end onwards is shifted by 4
# old_arr_end = 959 (the insertion point in our patched array)
# If orig_off > old_arr_end, then in the patched array, tensor info is at orig_off + 4

tensor_info_pos_patched = orig_off
if orig_off >= old_arr_end:
    tensor_info_pos_patched = orig_off + 4  # shifted by our insertion

print(f'Tensor info in patched file: byte {tensor_info_pos_patched}')

# ── Read all tensor offsets, adjust, and write back ──
# GGUF v3 tensor info layout (confirmed by gguf_reader.py _get_tensor_info_field):
#   name_len(uint64) + name + n_dims(uint32) + dims(uint64[]) + dtype(uint32) + offset(uint64)
tio = tensor_info_pos_patched
print(f'  First tensor at tio={tio}, raw bytes: {raw[tio:tio+16].hex()}')
tensor_data_offsets_orig = []
for i in range(tc2):
    if i < 3:
        print(f'  tensor[{i}] at tio={tio}')
    nl, = struct.unpack_from('<Q', raw, tio); tio += 8  # name_len
    tio += nl                                                  # skip name
    nd, = struct.unpack_from('<I', raw, tio); tio += 4        # n_dims
    tio += nd * 8                                              # skip dims
    tio += 4                                                   # skip dtype (uint32)
    toff, = struct.unpack_from('<Q', raw, tio); tio += 8      # offset (uint64)
    tensor_data_offsets_orig.append(toff)

# Verify we found all tensors
print(f'  Parsed {len(tensor_data_offsets_orig)} tensor info entries')
print(f'  Tensor info section: {tensor_info_pos_patched} - {tio} ({tio - tensor_info_pos_patched} bytes)')

# Note: tensor offsets are RELATIVE to the data section start, so we do NOT
# increment them. The 4 inserted bytes are in the metadata (before the data
# section), so the data section shifts as a whole but relative offsets stay
# the same. llama.cpp also enforces offset=0 for the first tensor.
print('  Tensor offsets unchanged (relative to data section start)')

# ── Write output ──
with open(DST, 'wb') as f:
    f.write(raw)

print(f'\nWritten: {DST}')
print(f'Size: {os.path.getsize(DST) / 1e9:.2f} GB (+4 bytes)')

# ── Verify ──
import gguf
r = gguf.GGUFReader(DST, 'r')
sec = r.fields.get('qwen35.rope.dimension_sections')
if sec and len(sec.contents()) == 4:
    print(f'\nVerification: sections={list(sec.contents())} ✅')
else:
    print(f'\nVerification FAILED: sections={list(sec.contents()) if sec else "MISSING"} ❌')

print(f'Tensors: {len(r.tensors)}')
