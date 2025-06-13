import os

def find_missing_files(folder_path):
    expected_files = {f"{str(i).zfill(6)}.png" for i in range(50000)}
    actual_files = set(os.listdir(folder_path))
    missing_files = sorted(expected_files - actual_files)
    return missing_files


if __name__ == '__main__':
    folder_path = '/data/MoE_exp/imagenet256/samples/DiT-S-2-8E2A2TP-Top_P_FreqT_woVAE_fp16-0400000-size-256-vae-mse-cfg-1.5-seed-0'
    missing_files = find_missing_files(folder_path)
    print(missing_files)
