"""
GEHIRN v0.5 - Central Orchestrator
Gerencia as 4 fases de execução com checkpointing, logging e controle de tempo.
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum

# Imports locais
from core.models import GEHIRN, ConstraintEncoder, PhysicsValidator
from core.physics import MultiDomainPhysicsLoss, EikonalConstraint
from core.utils import (
    setup_logging, save_checkpoint, load_checkpoint,
    save_config, load_config, save_metrics,
    normalize_constraints, denormalize_constraints,
    sdf_tensor_to_numpy, compute_sdf_statistics, compute_geometry_quality
)


class PhaseType(Enum):
    """Tipos de fases do pipeline."""
    KNOWLEDGE = "knowledge"
    TRAINING = "training"
    VALIDATION = "validation"
    PRODUCTION = "production"


@dataclass
class PhaseConfig:
    """Configuração de uma fase."""
    phase_type: PhaseType
    duration_hours: float
    enabled: bool
    checkpoint_interval_minutes: int
    
    def __post_init__(self):
        if not 1 <= self.duration_hours <= 24:
            raise ValueError(f"Duration must be 1-24 hours, got {self.duration_hours}")
        if not 1 <= self.checkpoint_interval_minutes <= 120:
            raise ValueError(f"Checkpoint interval must be 1-120 minutes, got {self.checkpoint_interval_minutes}")


@dataclass
class PhaseMetrics:
    """Métricas de uma fase."""
    phase_type: PhaseType
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    checkpoints_saved: int = 0
    status: str = "running"  # running, completed, failed, paused
    error_message: Optional[str] = None
    custom_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


class PhaseCheckpoint:
    """Gerencia checkpoints de uma fase."""
    
    def __init__(
        self,
        phase_type: PhaseType,
        checkpoint_dir: Path,
        interval_minutes: int = 15
    ):
        self.phase_type = phase_type
        self.checkpoint_dir = Path(checkpoint_dir) / phase_type.value
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval_minutes = interval_minutes
        self.last_checkpoint_time = time.time()
        self.checkpoint_count = 0
    
    def should_checkpoint(self) -> bool:
        """Verifica se é hora de fazer checkpoint."""
        elapsed_minutes = (time.time() - self.last_checkpoint_time) / 60
        return elapsed_minutes >= self.interval_minutes
    
    def save(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict] = None,
        data: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Salva checkpoint com timestamp.
        
        Args:
            model: Modelo PyTorch
            optimizer: Otimizador
            metrics: Métricas da fase
            data: Dados da fase (datasets, etc)
            metadata: Metadata adicional
            
        Returns:
            Caminho do checkpoint salvo
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"checkpoint_{self.checkpoint_count:05d}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint_dict = {
            'phase': self.phase_type.value,
            'checkpoint_number': self.checkpoint_count,
            'timestamp': timestamp,
            'metrics': metrics or {},
            'metadata': metadata or {},
        }
        
        if model is not None:
            checkpoint_dict['model_state'] = model.state_dict()
        
        if optimizer is not None:
            checkpoint_dict['optimizer_state'] = optimizer.state_dict()
        
        if data is not None:
            checkpoint_dict['data'] = data
        
        torch.save(checkpoint_dict, checkpoint_path)
        
        self.last_checkpoint_time = time.time()
        self.checkpoint_count += 1
        
        return checkpoint_path
    
    def load_latest(self, device: str = 'cuda') -> Optional[Dict]:
        """
        Carrega o checkpoint mais recente.
        
        Returns:
            Dict com conteúdo do checkpoint ou None
        """
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
        if not checkpoints:
            return None
        
        latest = checkpoints[-1]
        return torch.load(latest, map_location=device)
    
    def list_checkpoints(self) -> List[Path]:
        """Lista todos os checkpoints da fase."""
        return sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
    
    def cleanup(self, keep_last_n: int = 5):
        """Remove checkpoints antigos, mantendo os últimos N."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) > keep_last_n:
            for old_checkpoint in checkpoints[:-keep_last_n]:
                old_checkpoint.unlink()


class GEHIRN_Orchestrator:
    """
    Orquestrador central da GEHIRN v0.5.
    Gerencia as 4 fases: Knowledge → Training → Validation → Production
    
    Recursos:
    - Checkpointing automático a cada 15 minutos
    - Logging estruturado de todas as operações
    - Controle de tempo por fase (4-8 horas)
    - Recuperação de falhas via checkpoints
    - Métricas rastreáveis
    """
    
    def __init__(
        self,
        config_path: str = "./config/settings.yaml",
        phases_config_path: str = "./config/phases.yaml",
        materials_config_path: str = "./config/materials.yaml",
        device: str = "cuda"
    ):
        """
        Inicializa orquestrador.
        
        Args:
            config_path: Caminho para settings.yaml
            phases_config_path: Caminho para phases.yaml
            materials_config_path: Caminho para materials.yaml
            device: Device para PyTorch ('cuda' ou 'cpu')
        """
        self.device = device
        self.start_time = datetime.now()
        
        # Carregar configurações
        self.settings = self._load_yaml(config_path)
        self.phases_config = self._load_yaml(phases_config_path)
        self.materials = self._load_yaml(materials_config_path)
        
        # Setup de diretórios
        self.storage_root = Path(self.settings['storage']['root_dir'])
        self._setup_directories()
        
        # Setup de logging
        self.logger = setup_logging(
            log_dir=str(self.storage_root / 'logs'),
            name='gehirn_orchestrator'
        )
        
        self.logger.info(f"🚀 GEHIRN Orchestrator initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Storage root: {self.storage_root}")
        
        # Inicializar modelo
        self.model = None
        self.optimizer = None
        self.physics_loss = None
        
        # Checkpoints
        self.checkpoints = {}
        
        # Histórico de métricas
        self.metrics_history = []
        self.current_phase_metrics = None
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Carrega arquivo YAML."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_directories(self):
        """Cria estrutura de diretórios necessária."""
        dirs = [
            self.storage_root / 'models',
            self.storage_root / 'datasets' / 'knowledge',
            self.storage_root / 'logs',
            self.storage_root / 'artifacts' / 'designs',
            self.storage_root / 'artifacts' / 'meshes',
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def initialize_model(
        self,
        voxel_size: Optional[int] = None,
        latent_dim: Optional[int] = None
    ) -> GEHIRN:
        """
        Inicializa modelo GEHIRN.
        
        Args:
            voxel_size: Tamanho do voxel (padrão: do config)
            latent_dim: Dimensão latente (padrão: do config)
            
        Returns:
            Modelo inicializado e movido para device
        """
        voxel_size = voxel_size or self.settings['model']['voxel_size']
        latent_dim = latent_dim or self.settings['model']['latent_dim']
        
        self.model = GEHIRN(
            voxel_size=voxel_size,
            latent_dim=latent_dim,
            enable_physics_validation=self.settings['model']['enable_physics_validation']
        ).to(self.device)
        
        # Inicializar otimizador
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.phases_config['phases']['training']['hyperparameters']['learning_rate'],
            weight_decay=self.phases_config['phases']['training']['hyperparameters']['weight_decay']
        )
        
        # Inicializar loss multi-domínio
        self.physics_loss = MultiDomainPhysicsLoss(
            domains=self.settings['domains']['supported'],
            weights=self.settings['domains']['weights']
        )
        
        param_count = self.model.get_parameter_count()
        self.logger.info(f"Model initialized with {param_count['total']:,} parameters")
        self.logger.info(f"  - Constraint Encoder: {param_count['constraint_encoder']:,}")
        self.logger.info(f"  - Decoder: {param_count['decoder']:,}")
        self.logger.info(f"  - Physics Validator: {param_count['physics_validator']:,}")
        
        return self.model
    
    def start_phase(
        self,
        phase_type: PhaseType,
        duration_hours: float,
        **phase_specific_args
    ) -> 'PhaseExecutor':
        """
        Inicia uma fase com tempo limite e checkpointing automático.
        
        Args:
            phase_type: Tipo da fase
            duration_hours: Duração em horas
            **phase_specific_args: Argumentos específicos da fase
            
        Returns:
            PhaseExecutor para gerenciar a fase
        """
        phase_config = self.phases_config['phases'][phase_type.value]
        
        if not phase_config['enabled']:
            raise ValueError(f"Phase {phase_type.value} is disabled in config")
        
        # Validar duração
        if duration_hours not in phase_config['duration_options']:
            self.logger.warning(
                f"Duration {duration_hours}h not in options {phase_config['duration_options']}, "
                f"usando o mais próximo"
            )
            duration_hours = min(
                phase_config['duration_options'],
                key=lambda x: abs(x - duration_hours)
            )
        
        # Criar executor da fase
        executor = PhaseExecutor(
            orchestrator=self,
            phase_type=phase_type,
            duration_hours=duration_hours,
            phase_config=phase_config,
            device=self.device,
            logger=self.logger
        )
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Starting Phase: {phase_type.value.upper()}")
        self.logger.info(f"Duration: {duration_hours} hours")
        self.logger.info(f"Checkpoint interval: {phase_config['checkpoint']['interval_minutes']} minutes")
        self.logger.info(f"{'='*70}\n")
        
        self.current_phase_metrics = PhaseMetrics(
            phase_type=phase_type,
            start_time=datetime.now().isoformat()
        )
        
        return executor
    
    def end_phase(self, executor: 'PhaseExecutor') -> PhaseMetrics:
        """
        Finaliza uma fase e registra métricas.
        
        Args:
            executor: PhaseExecutor da fase concluída
            
        Returns:
            PhaseMetrics com informações da fase
        """
        self.current_phase_metrics.end_time = datetime.now().isoformat()
        self.current_phase_metrics.duration_seconds = executor.get_elapsed_seconds()
        self.current_phase_metrics.checkpoints_saved = executor.checkpoint_manager.checkpoint_count
        self.current_phase_metrics.status = "completed"
        self.current_phase_metrics.custom_metrics = executor.get_metrics()
        
        # Salvar métricas
        self.metrics_history.append(asdict(self.current_phase_metrics))
        save_metrics(
            self.metrics_history,
            str(self.storage_root / 'logs' / 'metrics_history.json')
        )
        
        self.logger.info(f"\nPhase {executor.phase_type.value} completed")
        self.logger.info(f"Duration: {self.current_phase_metrics.duration_seconds:.1f} seconds")
        self.logger.info(f"Checkpoints saved: {self.current_phase_metrics.checkpoints_saved}")
        self.logger.info(f"Status: {self.current_phase_metrics.status}")
        
        return self.current_phase_metrics
    
    def save_session(self, session_name: str) -> Path:
        """
        Salva sessão completa (modelo, config, métricas).
        
        Args:
            session_name: Nome da sessão
            
        Returns:
            Caminho da sessão salva
        """
        session_dir = self.storage_root / 'sessions' / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar modelo
        if self.model is not None:
            torch.save(
                self.model.state_dict(),
                session_dir / 'model_final.pt'
            )
        
        # Salvar configs
        save_config(self.settings, str(session_dir / 'settings.json'))
        save_config(self.phases_config, str(session_dir / 'phases.json'))
        
        # Salvar métricas
        save_metrics(self.metrics_history, str(session_dir / 'metrics.json'))
        
        # Salvar metadata
        metadata = {
            'session_name': session_name,
            'timestamp': datetime.now().isoformat(),
            'total_duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'device': self.device,
            'total_checkpoints': sum(
                metrics.get('checkpoints_saved', 0)
                for metrics in self.metrics_history
            ),
        }
        save_config(metadata, str(session_dir / 'metadata.json'))
        
        self.logger.info(f"Session saved to {session_dir}")
        return session_dir
    
    def load_session(self, session_path: str) -> bool:
        """
        Carrega sessão anterior.
        
        Args:
            session_path: Caminho da sessão
            
        Returns:
            True se carregou com sucesso
        """
        session_dir = Path(session_path)
        
        try:
            # Carregar modelo
            if (session_dir / 'model_final.pt').exists():
                self.initialize_model()
                state = torch.load(
                    session_dir / 'model_final.pt',
                    map_location=self.device
                )
                self.model.load_state_dict(state)
                self.logger.info(f"Model loaded from {session_path}")
            
            # Carregar métricas
            if (session_dir / 'metrics.json').exists():
                with open(session_dir / 'metrics.json', 'r') as f:
                    self.metrics_history = json.load(f)
                self.logger.info(f"Metrics loaded from {session_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load session: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status atual do orquestrador.
        """
        return {
            'elapsed_time': (datetime.now() - self.start_time).total_seconds(),
            'phases_completed': len(self.metrics_history),
            'total_checkpoints': sum(
                m.get('checkpoints_saved', 0) for m in self.metrics_history
            ),
            'model_initialized': self.model is not None,
            'device': self.device,
            'metrics': self.metrics_history,
        }


class PhaseExecutor:
    """
    Executor de uma fase individual.
    Gerencia tempo, checkpointing e métricas da fase.
    """
    
    def __init__(
        self,
        orchestrator: GEHIRN_Orchestrator,
        phase_type: PhaseType,
        duration_hours: float,
        phase_config: Dict,
        device: str,
        logger: Any
    ):
        self.orchestrator = orchestrator
        self.phase_type = phase_type
        self.duration_hours = duration_hours
        self.phase_config = phase_config
        self.device = device
        self.logger = logger
        
        self.start_time = time.time()
        self.end_time = None
        self.time_limit_seconds = duration_hours * 3600
        
        # Checkpoint manager
        checkpoint_interval = phase_config['checkpoint']['interval_minutes']
        self.checkpoint_manager = PhaseCheckpoint(
            phase_type=phase_type,
            checkpoint_dir=orchestrator.storage_root / 'checkpoints',
            interval_minutes=checkpoint_interval
        )
        
        self.metrics = {}
    
    def get_elapsed_seconds(self) -> float:
        """Retorna segundos decorridos."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def get_remaining_seconds(self) -> float:
        """Retorna segundos restantes."""
        elapsed = self.get_elapsed_seconds()
        return max(0, self.time_limit_seconds - elapsed)
    
    def should_continue(self) -> bool:
        """Verifica se ainda há tempo na fase."""
        return self.get_remaining_seconds() > 0
    
    def checkpoint_if_needed(
        self,
        model: Optional[nn.Module] = None,
        metrics: Optional[Dict] = None
    ) -> Optional[Path]:
        """
        Faz checkpoint se o intervalo tiver passado.
        """
        if self.checkpoint_manager.should_checkpoint():
            checkpoint_path = self.checkpoint_manager.save(
                model=model,
                optimizer=self.orchestrator.optimizer,
                metrics=metrics,
                metadata={'elapsed_hours': self.get_elapsed_seconds() / 3600}
            )
            self.logger.info(f"Checkpoint saved: {checkpoint_path.name}")
            return checkpoint_path
        return None
    
    def finalize(self) -> None:
        """Finaliza a fase."""
        self.end_time = time.time()
        self.checkpoint_manager.cleanup(keep_last_n=5)
        self.logger.info(f"Phase {self.phase_type.value} finalized")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas da fase."""
        return {
            'elapsed_seconds': self.get_elapsed_seconds(),
            'remaining_seconds': self.get_remaining_seconds(),
            'checkpoints_count': self.checkpoint_manager.checkpoint_count,
            **self.metrics
        }

