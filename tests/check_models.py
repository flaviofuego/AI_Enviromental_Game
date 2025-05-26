# check_models.py
import os
import numpy as np
from stable_baselines3 import PPO, DQN

def check_model_compatibility(model_path):
    """Check if a model can be loaded and what type it is"""
    print(f"\nChecking: {model_path}")
    
    if not os.path.exists(model_path):
        print("  ❌ File does not exist")
        return None
    
    # Try PPO first
    try:
        model = PPO.load(model_path)
        print("  ✅ Successfully loaded as PPO")
        
        # Check observation space
        obs_space = model.observation_space
        print(f"  📊 Observation space: {obs_space}")
        
        # Test with dummy observation
        if hasattr(obs_space, 'shape'):
            obs_dim = obs_space.shape[0]
            print(f"  🔢 Observation dimensions: {obs_dim}")
            
            # Determine model type based on dimensions
            if obs_dim == 21:
                model_type = "improved"
            elif obs_dim == 13:
                model_type = "original"
            else:
                model_type = f"unknown ({obs_dim} dims)"
            
            print(f"  🎯 Model type: {model_type}")
            
            # Test prediction
            dummy_obs = np.zeros(obs_dim, dtype=np.float32)
            try:
                action, _ = model.predict(dummy_obs, deterministic=True)
                print(f"  ✅ Test prediction successful: action = {action}")
                return {"type": "PPO", "model_type": model_type, "obs_dim": obs_dim, "working": True}
            except Exception as e:
                print(f"  ❌ Test prediction failed: {e}")
                return {"type": "PPO", "model_type": model_type, "obs_dim": obs_dim, "working": False}
        else:
            print("  ⚠️  Could not determine observation space shape")
            return {"type": "PPO", "model_type": "unknown", "obs_dim": "unknown", "working": False}
            
    except Exception as e:
        print(f"  ❌ Failed to load as PPO: {e}")
    
    # Try DQN
    try:
        model = DQN.load(model_path)
        print("  ✅ Successfully loaded as DQN")
        
        # Check observation space
        obs_space = model.observation_space
        print(f"  📊 Observation space: {obs_space}")
        
        # Test with dummy observation
        if hasattr(obs_space, 'shape'):
            obs_dim = obs_space.shape[0]
            print(f"  🔢 Observation dimensions: {obs_dim}")
            
            # Determine model type based on dimensions
            if obs_dim == 21:
                model_type = "improved"
            elif obs_dim == 13:
                model_type = "original"
            else:
                model_type = f"unknown ({obs_dim} dims)"
            
            print(f"  🎯 Model type: {model_type}")
            
            # Test prediction
            dummy_obs = np.zeros(obs_dim, dtype=np.float32)
            try:
                action, _ = model.predict(dummy_obs, deterministic=True)
                print(f"  ✅ Test prediction successful: action = {action}")
                return {"type": "DQN", "model_type": model_type, "obs_dim": obs_dim, "working": True}
            except Exception as e:
                print(f"  ❌ Test prediction failed: {e}")
                return {"type": "DQN", "model_type": model_type, "obs_dim": obs_dim, "working": False}
        else:
            print("  ⚠️  Could not determine observation space shape")
            return {"type": "DQN", "model_type": "unknown", "obs_dim": "unknown", "working": False}
            
    except Exception as e:
        print(f"  ❌ Failed to load as DQN: {e}")
    
    print("  ❌ Could not load with any algorithm")
    return None

def main():
    print("🔍 MODEL COMPATIBILITY CHECKER")
    print("=" * 50)
    
    # List of potential model paths
    model_paths = [
        # Improved models
        "improved_models/improved_air_hockey_final.zip",
        "improved_models/quick_model_final.zip",
        
        # Original models
        "models/air_hockey_ppo_final.zip",
    ]
    
    # Add any other .zip files found in directories
    if os.path.exists("..improved_models"):
        for file in os.listdir("improved_models"):
            if file.endswith(".zip") and file not in [os.path.basename(p) for p in model_paths]:
                model_paths.append(os.path.join("..improved_models", file))
    
    if os.path.exists("..models"):
        for file in os.listdir("..models"):
            if file.endswith(".zip") and file not in [os.path.basename(p) for p in model_paths]:
                model_paths.append(os.path.join("..models", file))
    
    # Check current directory for .zip files
    for file in os.listdir("."):
        if file.endswith(".zip") and file not in [os.path.basename(p) for p in model_paths]:
            model_paths.append(file)
    
    working_models = []
    broken_models = []
    
    for model_path in model_paths:
        result = check_model_compatibility(model_path)
        if result:
            if result["working"]:
                working_models.append((model_path, result))
            else:
                broken_models.append((model_path, result))
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    if working_models:
        print(f"\n✅ WORKING MODELS ({len(working_models)}):")
        for model_path, info in working_models:
            print(f"  • {model_path}")
            print(f"    - Algorithm: {info['type']}")
            print(f"    - Type: {info['model_type']}")
            print(f"    - Dimensions: {info['obs_dim']}")
    
    if broken_models:
        print(f"\n❌ BROKEN MODELS ({len(broken_models)}):")
        for model_path, info in broken_models:
            print(f"  • {model_path}")
            print(f"    - Algorithm: {info['type']}")
            print(f"    - Type: {info['model_type']}")
            print(f"    - Dimensions: {info['obs_dim']}")
    
    if not working_models and not broken_models:
        print("\n❌ NO MODELS FOUND")
        print("Please train a model first using:")
        print("  python quick_train.py")
        print("  or")
        print("  python improved_training_system.py")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    if working_models:
        # Find best model
        improved_models = [m for m in working_models if m[1]["model_type"] == "improved"]
        if improved_models:
            best_model = improved_models[0][0]
            print(f"  • Use improved model: {best_model}")
        else:
            best_model = working_models[0][0]
            print(f"  • Use available model: {best_model}")
        
        print(f"  • Run: python main_improved.py")
        print(f"  • Select option 2 for auto-detection")
    else:
        print("  • Train a new model first:")
        print("    python setup_simple.py")
        print("    python quick_train.py")

if __name__ == "__main__":
    main() 