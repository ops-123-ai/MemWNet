import argparse

parser = argparse.ArgumentParser(description="Args of this repo.")

parser.add_argument("--rate", default=1, type=int)
parser.add_argument("--stage", default=7, type=int)
parser.add_argument("--alpha", default=9, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--device", default="0")
parser.add_argument("--time", default=0, type=int)
parser.add_argument("--patch_size", default=128, type=int)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--save", default=False)
parser.add_argument("--norm", default=0, type=int)
parser.add_argument("--save_path", default=f"./models_bipolar")
parser.add_argument('--matrix_dir', type=str,
                    default='sampling_matrix', help='sampling matrix directory')
parser.add_argument("--folder")
parser.add_argument("--my_state_dict")
parser.add_argument("--my_log")
parser.add_argument("--my_info")

para = parser.parse_args()

if para.device == "cpu":
    para.device = "cpu"
else:
    para.device = f"cuda:{para.device}"

para.folder = f"{para.save_path}/{str(int(para.rate))}/"
para.my_state_dict = f"{para.folder}/step_num_state_dict.pth"
para.my_log = f"{para.folder}/step_num_log.txt"
para.my_info = f"{para.folder}/step_num_info.pth"