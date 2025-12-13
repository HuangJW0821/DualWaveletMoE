import json
import matplotlib.pyplot as plt

def extract_loss_log(json_file_paths):
    steps = []
    losses = []
    max_step = 0

    for path in json_file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            log_history = data.get('log_history', [])

            for entry in log_history:
                if 'step' in entry and 'loss' in entry:
                    steps.append(entry['step'] + max_step)
                    losses.append(entry['loss'])
            
            if not steps:
                print("STEP or LOSS NOT FOUND")
                return
            
        except FileNotFoundError:
            print(f"ERROR: file not found {path}")
        except json.JSONDecodeError:
            print(f"ERROR: {path} is not json file")
        except Exception as e:
            print(f"ERROR: {str(e)}")

        max_step = max(steps)
    
    return steps, losses

def plot_loss_from_trainer_state(json_file_paths, output_image_path=None):

    steps, losses = extract_loss_log(json_file_paths)

    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, 'b-', linewidth=1.5, alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
        
    if len(steps) > 20:
        plt.xticks(rotation=45)
        
    plt.tight_layout()
        
    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved at: {output_image_path}")
        
    # 显示图片
    plt.show()
        
    print(f"Total {len(steps)} data points")
    print(f"Loss range: {min(losses):.4f} - {max(losses):.4f}")


if __name__ == "__main__":

    json_files = [
        "/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/WaveletMoE_multivariate/logs/uni_50M_syn_from_scratch/checkpoint-50000/trainer_state.json"
    ]
    
    output_file = "/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/WaveletMoE_multivariate/figs/uni_50M_syn_from_scratch/loss_curve"

    plot_loss_from_trainer_state(json_files, output_file)