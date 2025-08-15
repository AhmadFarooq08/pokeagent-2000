#!/usr/bin/env python3
"""
Complete Training Pipeline
Orchestrates the entire Metamon training process from data download to competition
"""

import os
import sys
import json
import time
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingOrchestrator:
    """Orchestrates the complete training pipeline"""
    
    def __init__(self, 
                 work_dir: str = ".",
                 debug_mode: bool = False,
                 skip_download: bool = False,
                 skip_bc: bool = False,
                 skip_rl: bool = False,
                 skip_self_play: bool = False):
        
        self.work_dir = Path(work_dir).absolute()
        self.debug_mode = debug_mode
        self.skip_download = skip_download
        self.skip_bc = skip_bc
        self.skip_rl = skip_rl
        self.skip_self_play = skip_self_play
        
        # Create directories
        self.create_directories()
        
        # Training state
        self.training_state = {
            'start_time': time.time(),
            'completed_stages': [],
            'current_stage': None,
            'checkpoints': {},
            'debug_mode': debug_mode
        }
        
        # Save state file
        self.state_file = self.work_dir / "training_state.json"
        self.save_state()
    
    def create_directories(self):
        """Create all necessary directories"""
        dirs = [
            'data/pokechamp_raw',
            'data/processed_replays', 
            'data/self_play',
            'checkpoints',
            'logs',
            'teams'
        ]
        
        for dir_path in dirs:
            (self.work_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Created directory structure")
    
    def save_state(self):
        """Save training state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.training_state, f, indent=2)
    
    def load_state(self):
        """Load training state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.training_state = json.load(f)
    
    def run_command(self, cmd: List[str], description: str, cwd: str = None) -> bool:
        """Run a command and log output"""
        logger.info(f"Starting: {description}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            if cwd is None:
                cwd = str(self.work_dir)
            
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"✅ {description} completed successfully")
            
            # Log output if not too long
            if len(result.stdout) < 2000:
                logger.info(f"Output: {result.stdout}")
            else:
                logger.info(f"Output: {result.stdout[:1000]}... (truncated)")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {description} failed")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            return False
        
        except Exception as e:
            logger.error(f"❌ {description} failed with exception: {e}")
            return False
    
    def stage_1_data_download(self) -> bool:
        """Stage 1: Download and process dataset"""
        if self.skip_download:
            logger.info("⏭️ Skipping data download (--skip-download)")
            return True
        
        self.training_state['current_stage'] = 'data_download'
        self.save_state()
        
        logger.info("🗂️ STAGE 1: Data Download and Processing")
        
        # Download dataset
        cmd = [
            sys.executable, "scripts/download_dataset.py",
            "--output-dir", "data",
            "--filter-gen9ou",
            "--validate"
        ]
        
        if self.debug_mode:
            cmd.extend(["--analyze"])  # Add analysis in debug mode
        
        if not self.run_command(cmd, "Download 3.5M replay dataset"):
            return False
        
        # Reconstruct replays
        cmd = [
            sys.executable, "scripts/reconstruct_replays.py",
            "--input-dir", "data/pokechamp_raw",
            "--output-dir", "data/processed_replays",
            "--format-filter", "gen9ou"
        ]
        
        if self.debug_mode:
            cmd.extend(["--max-replays", "1000"])  # Limit for debug
        
        if not self.run_command(cmd, "Reconstruct first-person replay views"):
            return False
        
        self.training_state['completed_stages'].append('data_download')
        self.save_state()
        
        logger.info("✅ Stage 1 completed: Data download and processing")
        return True
    
    def stage_2_behavior_cloning(self) -> bool:
        """Stage 2: Behavior cloning training"""
        if self.skip_bc:
            logger.info("⏭️ Skipping behavior cloning (--skip-bc)")
            return True
        
        self.training_state['current_stage'] = 'behavior_cloning'
        self.save_state()
        
        logger.info("🧠 STAGE 2: Behavior Cloning Training")
        
        cmd = [
            sys.executable, "scripts/train_bc.py",
            "--data-dir", "data/processed_replays",
            "--output-dir", "checkpoints"
        ]
        
        if self.debug_mode:
            cmd.extend(["--debug"])
        
        if not self.run_command(cmd, "Behavior cloning training"):
            return False
        
        # Check for BC checkpoint
        bc_checkpoint = self.work_dir / "checkpoints" / "best_bc_model.pt"
        if not bc_checkpoint.exists():
            logger.error("❌ BC checkpoint not found")
            return False
        
        self.training_state['checkpoints']['bc_model'] = str(bc_checkpoint)
        self.training_state['completed_stages'].append('behavior_cloning')
        self.save_state()
        
        logger.info("✅ Stage 2 completed: Behavior cloning")
        return True
    
    def stage_3_offline_rl(self) -> bool:
        """Stage 3: Offline RL training"""
        if self.skip_rl:
            logger.info("⏭️ Skipping offline RL (--skip-rl)")
            return True
        
        self.training_state['current_stage'] = 'offline_rl'
        self.save_state()
        
        logger.info("🎯 STAGE 3: Offline RL Training")
        
        # Get BC checkpoint
        bc_checkpoint = self.training_state['checkpoints'].get('bc_model')
        if not bc_checkpoint:
            logger.error("❌ BC checkpoint not available")
            return False
        
        cmd = [
            sys.executable, "scripts/train_rl.py",
            "--data-dir", "data/processed_replays",
            "--bc-checkpoint", bc_checkpoint,
            "--output-dir", "checkpoints"
        ]
        
        if self.debug_mode:
            cmd.extend(["--debug"])
        
        if not self.run_command(cmd, "Offline RL training"):
            return False
        
        # Check for RL checkpoint
        rl_checkpoint = self.work_dir / "checkpoints" / "best_rl_model.pt"
        if not rl_checkpoint.exists():
            logger.error("❌ RL checkpoint not found")
            return False
        
        self.training_state['checkpoints']['rl_model'] = str(rl_checkpoint)
        self.training_state['completed_stages'].append('offline_rl')
        self.save_state()
        
        logger.info("✅ Stage 3 completed: Offline RL")
        return True
    
    def stage_4_self_play(self) -> bool:
        """Stage 4: Self-play data generation"""
        if self.skip_self_play:
            logger.info("⏭️ Skipping self-play (--skip-self-play)")
            return True
        
        self.training_state['current_stage'] = 'self_play'
        self.save_state()
        
        logger.info("🥊 STAGE 4: Self-Play Data Generation")
        
        # Get RL checkpoint
        rl_checkpoint = self.training_state['checkpoints'].get('rl_model')
        if not rl_checkpoint:
            logger.error("❌ RL checkpoint not available")
            return False
        
        cmd = [
            sys.executable, "scripts/self_play.py",
            "--model-path", rl_checkpoint,
            "--output-dir", "data/self_play",
            "--num-games", "10000" if self.debug_mode else "100000"
        ]
        
        if self.debug_mode:
            cmd.extend(["--debug"])
        
        if not self.run_command(cmd, "Self-play data generation"):
            return False
        
        self.training_state['completed_stages'].append('self_play')
        self.save_state()
        
        logger.info("✅ Stage 4 completed: Self-play data generation")
        return True
    
    def stage_5_finalization(self) -> bool:
        """Stage 5: Model finalization and optimization"""
        self.training_state['current_stage'] = 'finalization'
        self.save_state()
        
        logger.info("🏁 STAGE 5: Model Finalization")
        
        # Copy best model to final location
        rl_checkpoint = self.training_state['checkpoints'].get('rl_model')
        if not rl_checkpoint:
            logger.error("❌ No model checkpoint available")
            return False
        
        final_checkpoint = self.work_dir / "checkpoints" / "final_model.pt"
        
        try:
            import shutil
            shutil.copy2(rl_checkpoint, final_checkpoint)
            logger.info(f"✅ Final model saved to {final_checkpoint}")
        except Exception as e:
            logger.error(f"❌ Failed to copy final model: {e}")
            return False
        
        self.training_state['checkpoints']['final_model'] = str(final_checkpoint)
        self.training_state['completed_stages'].append('finalization')
        self.training_state['end_time'] = time.time()
        self.save_state()
        
        # Generate training summary
        self.generate_summary()
        
        logger.info("✅ Stage 5 completed: Model finalization")
        return True
    
    def generate_summary(self):
        """Generate training summary"""
        summary = {
            'training_completed': True,
            'total_time_hours': (self.training_state['end_time'] - self.training_state['start_time']) / 3600,
            'completed_stages': self.training_state['completed_stages'],
            'checkpoints': self.training_state['checkpoints'],
            'debug_mode': self.debug_mode,
            'completion_date': datetime.now().isoformat()
        }
        
        summary_file = self.work_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Training summary saved to {summary_file}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("🎉 TRAINING PIPELINE COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Total time: {summary['total_time_hours']:.1f} hours")
        logger.info(f"Stages completed: {', '.join(summary['completed_stages'])}")
        logger.info(f"Final model: {summary['checkpoints']['final_model']}")
        logger.info("=" * 60)
        logger.info("🚀 Ready for competition!")
        logger.info("=" * 60)
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete training pipeline"""
        logger.info("🚀 Starting complete Metamon training pipeline")
        logger.info(f"Working directory: {self.work_dir}")
        logger.info(f"Debug mode: {self.debug_mode}")
        
        stages = [
            ("Data Download", self.stage_1_data_download),
            ("Behavior Cloning", self.stage_2_behavior_cloning),
            ("Offline RL", self.stage_3_offline_rl),
            ("Self-Play", self.stage_4_self_play),
            ("Finalization", self.stage_5_finalization)
        ]
        
        for stage_name, stage_func in stages:
            logger.info(f"\n{'='*20} {stage_name.upper()} {'='*20}")
            
            if not stage_func():
                logger.error(f"❌ Pipeline failed at stage: {stage_name}")
                return False
        
        logger.info("\n🎉 Complete training pipeline finished successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Complete Metamon Training Pipeline")
    parser.add_argument("--work-dir", default=".", help="Working directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode (reduced data/epochs)")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--skip-bc", action="store_true", help="Skip behavior cloning")
    parser.add_argument("--skip-rl", action="store_true", help="Skip offline RL")
    parser.add_argument("--skip-self-play", action="store_true", help="Skip self-play")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = TrainingOrchestrator(
        work_dir=args.work_dir,
        debug_mode=args.debug,
        skip_download=args.skip_download,
        skip_bc=args.skip_bc,
        skip_rl=args.skip_rl,
        skip_self_play=args.skip_self_play
    )
    
    # Resume if requested
    if args.resume:
        orchestrator.load_state()
        logger.info("📂 Loaded previous training state")
    
    # Run pipeline
    try:
        success = orchestrator.run_complete_pipeline()
        
        if success:
            logger.info("🎯 Training pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Training pipeline failed!")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("⏸️ Training interrupted by user")
        logger.info("💾 State saved - use --resume to continue")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"💥 Training pipeline crashed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()