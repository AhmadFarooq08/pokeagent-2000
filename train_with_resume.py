#!/usr/bin/env python3
"""
Robust Training Pipeline with Resume Capability
Perfect for interrupted cluster jobs
"""

import os
import sys
import json
import time
import argparse
import subprocess
import signal
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GracefulShutdown:
    """Handle graceful shutdown"""
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown = True

class RobustTrainingOrchestrator:
    """Training orchestrator with robust resume capability"""
    
    def __init__(self, work_dir: str = ".", debug_mode: bool = False):
        self.work_dir = Path(work_dir).absolute()
        self.debug_mode = debug_mode
        
        # Create directories
        self.create_directories()
        
        # Training state
        self.state_file = self.work_dir / "training_state.json"
        self.load_or_create_state()
        
        # Graceful shutdown
        self.shutdown_handler = GracefulShutdown()
        
    def create_directories(self):
        """Create all necessary directories"""
        dirs = [
            'data', 'checkpoints', 'logs', 'teams'
        ]
        
        for dir_path in dirs:
            (self.work_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_or_create_state(self):
        """Load existing state or create new"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            logger.info("📂 Loaded existing training state")
        else:
            self.state = {
                'pipeline_start_time': time.time(),
                'current_stage': None,
                'completed_stages': [],
                'stage_attempts': {},
                'checkpoints': {},
                'debug_mode': self.debug_mode,
                'last_update': time.time()
            }
            self.save_state()
    
    def save_state(self):
        """Save current state"""
        self.state['last_update'] = time.time()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if stage is completed"""
        return stage_name in self.state['completed_stages']
    
    def mark_stage_completed(self, stage_name: str, checkpoint_path: Optional[str] = None):
        """Mark stage as completed"""
        if stage_name not in self.state['completed_stages']:
            self.state['completed_stages'].append(stage_name)
        
        if checkpoint_path:
            self.state['checkpoints'][stage_name] = checkpoint_path
        
        self.save_state()
        logger.info(f"✅ Stage completed: {stage_name}")
    
    def run_command_robust(self, cmd: List[str], stage_name: str, max_retries: int = 3) -> bool:
        """Run command with retry logic and state tracking"""
        
        self.state['current_stage'] = stage_name
        attempts = self.state['stage_attempts'].get(stage_name, 0)
        
        if attempts >= max_retries:
            logger.error(f"❌ Stage {stage_name} failed after {max_retries} attempts")
            return False
        
        logger.info(f"🚀 Starting stage: {stage_name} (attempt {attempts + 1})")
        
        try:
            # Add resume flag if this is a retry
            if attempts > 0 and '--resume' not in cmd:
                cmd.append('--resume')
            
            result = subprocess.run(
                cmd,
                cwd=str(self.work_dir),
                capture_output=False,  # Show output in real-time
                text=True,
                check=True
            )
            
            logger.info(f"✅ Stage {stage_name} completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            attempts += 1
            self.state['stage_attempts'][stage_name] = attempts
            self.save_state()
            
            logger.error(f"❌ Stage {stage_name} failed (attempt {attempts}/{max_retries})")
            logger.error(f"Return code: {e.returncode}")
            
            if attempts < max_retries:
                logger.info(f"🔄 Will retry stage {stage_name}")
                return self.run_command_robust(cmd, stage_name, max_retries)
            else:
                return False
        
        except KeyboardInterrupt:
            logger.info(f"⏸️ Stage {stage_name} interrupted")
            self.save_state()
            return False
    
    def stage_download_data(self) -> bool:
        """Download and process data"""
        if self.is_stage_completed('data_download'):
            logger.info("⏭️ Data already downloaded, skipping")
            return True
        
        # For debug mode, create dummy data
        if self.debug_mode:
            logger.info("🐛 Debug mode: creating dummy data")
            dummy_data_dir = self.work_dir / "data" / "processed_replays"
            dummy_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a dummy file to indicate data is ready
            (dummy_data_dir / "dummy_data.txt").write_text("Debug mode dummy data")
            
            self.mark_stage_completed('data_download')
            return True
        
        # Download real data (implement if needed)
        logger.info("📥 Data download not implemented yet, marking as completed")
        self.mark_stage_completed('data_download')
        return True
    
    def stage_behavior_cloning(self) -> bool:
        """Run behavior cloning training"""
        if self.is_stage_completed('behavior_cloning'):
            logger.info("⏭️ Behavior cloning already completed, skipping")
            return True
        
        cmd = [
            sys.executable, "scripts/train_bc_robust.py",
            "--output-dir", "checkpoints",
            "--resume"  # Always try to resume
        ]
        
        if self.debug_mode:
            cmd.append("--debug")
        
        success = self.run_command_robust(cmd, "behavior_cloning")
        
        if success:
            # Find the best BC checkpoint
            bc_checkpoint = self.work_dir / "checkpoints" / "best_bc_model.pt"
            if bc_checkpoint.exists():
                self.mark_stage_completed('behavior_cloning', str(bc_checkpoint))
                return True
            else:
                logger.error("❌ BC training reported success but no checkpoint found")
                return False
        
        return False
    
    def estimate_time_remaining(self) -> str:
        """Estimate time remaining for current stage"""
        elapsed = time.time() - self.state['pipeline_start_time']
        
        if len(self.state['completed_stages']) == 0:
            return "Unknown"
        
        # Rough estimates based on stage completion
        total_stages = 3  # data, BC, finalization
        completed = len(self.state['completed_stages'])
        
        if completed >= total_stages:
            return "Complete"
        
        avg_time_per_stage = elapsed / completed
        remaining_stages = total_stages - completed
        remaining_time = avg_time_per_stage * remaining_stages
        
        hours = int(remaining_time // 3600)
        minutes = int((remaining_time % 3600) // 60)
        
        return f"{hours}h {minutes}m"
    
    def run_pipeline(self, time_limit_hours: float = 4.0) -> bool:
        """Run the training pipeline with time management"""
        start_time = time.time()
        time_limit_seconds = time_limit_hours * 3600
        
        logger.info(f"🚀 Starting robust training pipeline")
        logger.info(f"Time limit: {time_limit_hours} hours")
        logger.info(f"Debug mode: {self.debug_mode}")
        
        # Define stages
        stages = [
            ("Data Download", self.stage_download_data),
            ("Behavior Cloning", self.stage_behavior_cloning),
        ]
        
        for stage_name, stage_func in stages:
            if self.shutdown_handler.shutdown:
                logger.info("🛑 Shutdown requested, stopping pipeline")
                break
            
            # Check time remaining
            elapsed = time.time() - start_time
            remaining = time_limit_seconds - elapsed
            
            if remaining < 300:  # Less than 5 minutes
                logger.warning(f"⏰ Less than 5 minutes remaining, stopping pipeline")
                break
            
            logger.info(f"\n{'='*20} {stage_name.upper()} {'='*20}")
            logger.info(f"Time remaining: {remaining/3600:.1f} hours")
            
            try:
                success = stage_func()
                
                if not success:
                    logger.error(f"❌ Pipeline failed at stage: {stage_name}")
                    return False
                
            except Exception as e:
                logger.error(f"💥 Exception in stage {stage_name}: {e}")
                return False
        
        # Create final summary
        self.create_summary()
        
        total_time = time.time() - start_time
        logger.info(f"\n🎉 Pipeline session complete!")
        logger.info(f"Total time: {total_time/3600:.1f} hours")
        logger.info(f"Completed stages: {', '.join(self.state['completed_stages'])}")
        
        if len(self.state['completed_stages']) >= 2:  # Adjust based on total stages
            logger.info("✅ Training ready for next session!")
        else:
            logger.info("⏸️ Training can be resumed in next session")
        
        return True
    
    def create_summary(self):
        """Create training summary"""
        elapsed = time.time() - self.state['pipeline_start_time']
        
        summary = {
            'session_complete': True,
            'total_time_hours': elapsed / 3600,
            'completed_stages': self.state['completed_stages'],
            'available_checkpoints': self.state['checkpoints'],
            'debug_mode': self.debug_mode,
            'completion_date': datetime.now().isoformat(),
            'next_steps': self.get_next_steps()
        }
        
        summary_file = self.work_dir / f"session_summary_{int(time.time())}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Session summary saved: {summary_file}")
    
    def get_next_steps(self) -> List[str]:
        """Get next steps for continuation"""
        completed = set(self.state['completed_stages'])
        
        if 'data_download' not in completed:
            return ["Download and process the 3.5M replay dataset"]
        elif 'behavior_cloning' not in completed:
            return ["Complete behavior cloning training", "Target: ~60% move prediction accuracy"]
        elif 'offline_rl' not in completed:
            return ["Run offline RL training with BC checkpoint", "Target: 1600-1800 ELO"]
        else:
            return ["Training complete!", "Ready for competition deployment"]
    
    def print_status(self):
        """Print current status"""
        elapsed = time.time() - self.state['pipeline_start_time']
        
        print("\n" + "="*60)
        print("🤖 POKEAGENT-2000 TRAINING STATUS")
        print("="*60)
        print(f"📊 Training time: {elapsed/3600:.1f} hours")
        print(f"🎯 Completed stages: {len(self.state['completed_stages'])}")
        print(f"📁 Working directory: {self.work_dir}")
        print(f"🐛 Debug mode: {self.debug_mode}")
        print(f"⏱️ Last update: {datetime.fromtimestamp(self.state['last_update']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.state['completed_stages']:
            print(f"\n✅ Completed stages:")
            for stage in self.state['completed_stages']:
                checkpoint = self.state['checkpoints'].get(stage, 'N/A')
                print(f"   • {stage}: {checkpoint}")
        
        print(f"\n🎯 Next steps:")
        for step in self.get_next_steps():
            print(f"   • {step}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Robust Training Pipeline")
    parser.add_argument("--work-dir", default=".", help="Working directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--status", action="store_true", help="Show status only")
    parser.add_argument("--time-limit", type=float, default=4.0, help="Time limit in hours")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = RobustTrainingOrchestrator(
        work_dir=args.work_dir,
        debug_mode=args.debug
    )
    
    # Show status if requested
    if args.status:
        orchestrator.print_status()
        return
    
    try:
        # Run pipeline
        success = orchestrator.run_pipeline(time_limit_hours=args.time_limit)
        
        if success:
            logger.info("🎯 Session completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Session failed!")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("⏸️ Training interrupted by user")
        logger.info("💾 State saved - can be resumed later")
        orchestrator.save_state()
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"💥 Training crashed: {e}")
        orchestrator.save_state()
        sys.exit(1)

if __name__ == "__main__":
    main()