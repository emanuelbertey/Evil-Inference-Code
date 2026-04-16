import struct

def read_keys():
    filepath = "xlstm/testbin_model/test_data.bin"
    with open(filepath, 'rb') as f:
        # X shape
        x_shape_len = struct.unpack('<I', f.read(4))[0]
        for _ in range(x_shape_len): f.read(4)
        x_data_len = struct.unpack('<I', f.read(4))[0]
        f.read(x_data_len)

        # Y shape
        y_shape_len = struct.unpack('<I', f.read(4))[0]
        for _ in range(y_shape_len): f.read(4)
        y_data_len = struct.unpack('<I', f.read(4))[0]
        f.read(y_data_len)

        # State dict
        num_tensors = struct.unpack('<I', f.read(4))[0]
        for _ in range(num_tensors):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            shape_len = struct.unpack('<I', f.read(4))[0]
            for _ in range(shape_len): f.read(4)
            data_len = struct.unpack('<I', f.read(4))[0]
            f.read(data_len)
            print(name)

if __name__ == "__main__":
    read_keys()
