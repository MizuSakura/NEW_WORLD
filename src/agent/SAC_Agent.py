# my_project/src/agent/SAC_Agent.py
 
import torch
import torch.nn.functional as F
from pathlib import Path
 
from src.agent.network import Actor, Critic
from src.agent.replaybuffer_manager import ReplayBufferManager
from src.utils.logger_pyarrow import MetricLogger
 
 
class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent
    ------------------------------------------------------------
    ออกแบบให้ยืดหยุ่นสูงสำหรับงานวิจัย/ Simulation / Control system
 
    - รองรับ Actor แบบ simple/advanced
    - รองรับ Critic แบบ simple/advanced + encoder=True/False
    - Twin Critic + Target Critic
    - Fixed alpha
    - Replay Buffer มาตรฐาน
    - ระบบ Auto-save + Resume training
 
    การสร้าง Agent มี 2 วิธี:
      1. จาก config โดยตรง (แนะนำ):
            agent = SACAgent.from_config(rl_cfg, state_dim, action_dim,
                                         min_action, max_action)
      2. ระบุ parameter เองทีละตัว (backward-compatible):
            agent = SACAgent(state_dim, action_dim, ...)
    """
 
    def __init__(self,
                 state_dim, action_dim,
                 min_action, max_action,
                 lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 replay_capacity=100000,
                 buffer_type="nstep_per",
                 n_step=3,
                 per_alpha=0.6,
                 per_beta=0.4,
                 device='cuda',
                 logger_status=False,
 
                 # -------- Actor Options ----------
                 simple_layers_actor=2,
                 simple_hidden_actor=256,
                 advanced_hidden_size_actor=None,
 
                 # -------- Critic Options ---------
                 simple_layers_critic=2,
                 simple_hidden_critic=256,
                 advanced_hidden_sizes_critic=None,
                 critic_encoder=False,
 
                 logger_path="logs/agent/RC_Tank",
                 file_name_log=None,
                 ):
 
        # ------------------------------------------------------------
        # Device
        # ------------------------------------------------------------
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print("[SACAgent] Using device:", self.device)
 
        # =====================================================================
        # Construct Actor
        # =====================================================================
        self.actor = Actor(
            state_dim, action_dim,
            min_action, max_action,
            simple_layers=simple_layers_actor,
            simple_hidden=simple_hidden_actor,
            advanced_hidden_sizes=advanced_hidden_size_actor
        ).to(self.device)
 
        # =====================================================================
        # Construct Critic (Main & Target) — Fully matching Architectures
        # =====================================================================
        self.critic = Critic(
            state_dim, action_dim,
            simple_layers=simple_layers_critic,
            simple_hidden=simple_hidden_critic,
            advanced_hidden_sizes=advanced_hidden_sizes_critic,
            use_encoder=critic_encoder
        ).to(self.device)
 
        self.target_critic = Critic(
            state_dim, action_dim,
            simple_layers=simple_layers_critic,
            simple_hidden=simple_hidden_critic,
            advanced_hidden_sizes=advanced_hidden_sizes_critic,
            use_encoder=critic_encoder
        ).to(self.device)
 
        self.target_critic.load_state_dict(self.critic.state_dict())
 
        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
 
        # =====================================================================
        # Replay Buffer — อ่านจาก config ทั้งหมด ไม่ hardcode
        # =====================================================================
        self.replay_buffer = ReplayBufferManager(
            buffer_type=buffer_type,
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=replay_capacity,
            n_step=n_step,
            gamma=gamma,
            alpha=per_alpha,
            beta=per_beta,
            device=self.device
        )
 
        # Hyperparameters
        self.gamma = gamma
        self.tau   = tau
        self.alpha = alpha
 
        # Store critic config for save/load
        self.simple_layers_critic         = simple_layers_critic
        self.simple_hidden_critic         = simple_hidden_critic
        self.advanced_hidden_sizes_critic = advanced_hidden_sizes_critic
        self.critic_encoder               = critic_encoder
 
        # Actor config (for save/load)
        self.simple_layers_actor        = simple_layers_actor
        self.simple_hidden_actor        = simple_hidden_actor
        self.advanced_hidden_size_actor = advanced_hidden_size_actor
 
        # Logger
        self.logger_path   = logger_path
        self.file_name_log = file_name_log
        self.logger        = MetricLogger(
            folder=self.logger_path,
            filename=self.file_name_log,
            auto_increment=True
        )
        self.logger_status = logger_status
        self.action_log    = None
 
    # ======================================================================
    # Factory: สร้าง SACAgent จาก RLConfig (pydantic) โดยตรง
    # ======================================================================
    @classmethod
    def from_config(cls, rl_cfg, state_dim: int, action_dim: int,
                    min_action, max_action,
                    device: str = "cuda",
                    logger_status: bool = False) -> "SACAgent":
        """
        สร้าง SACAgent จาก RLConfig object (จาก rl_schema.py)
 
        Parameters
        ----------
        rl_cfg      : RLConfig  (pydantic model จาก rl_loader.get_rl_config())
        state_dim   : int
        action_dim  : int
        min_action  : np.ndarray | list
        max_action  : np.ndarray | list
        device      : str  "cuda" | "cpu"
        logger_status : bool  เปิด/ปิด logging
 
        Example
        -------
        from src.API.src_api.rl_loader import get_rl_config
        from src.agent.SAC_Agent import SACAgent
        import numpy as np
 
        rl_cfg = get_rl_config()
        agent  = SACAgent.from_config(
            rl_cfg,
            state_dim  = rl_cfg.state.state_dim,
            action_dim = 1,
            min_action = np.array([0.0]),
            max_action = np.array([rl_cfg.state.level_max]),
        )
        """
        sac = rl_cfg.sac
        log = rl_cfg.logger
 
        return cls(
            state_dim    = state_dim,
            action_dim   = action_dim,
            min_action   = min_action,
            max_action   = max_action,
 
            # SAC hyperparams
            lr              = sac.learning_rate,
            gamma           = sac.gamma,
            tau             = sac.tau,
            alpha           = sac.alpha,
            replay_capacity = sac.replay_capacity,
            buffer_type     = sac.buffer_type,
            n_step          = sac.n_step,
            per_alpha       = sac.per_alpha,
            per_beta        = sac.per_beta,
 
            # Actor
            simple_layers_actor      = sac.actor.layers,
            simple_hidden_actor      = sac.actor.hidden,
            advanced_hidden_size_actor = None,
 
            # Critic
            simple_layers_critic         = sac.critic.layers,
            simple_hidden_critic         = sac.critic.hidden,
            advanced_hidden_sizes_critic = None,
            critic_encoder               = sac.critic.encoder,
 
            # Logger
            logger_path   = log.agent_folder,
            file_name_log = log.agent_filename,
            logger_status = logger_status,
 
            device = device,
        )
 
    # ======================================================================
    # Action Selection
    # ======================================================================
    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
 
        if deterministic:
            mean, _ = self.actor.forward(state)
            y = torch.tanh(mean)
            action = self.actor.min_action + (y + 1) * 0.5 * (self.actor.max_action - self.actor.min_action)
        else:
            action, _ = self.actor.sample(state)
        self.action_log = action
        return action.cpu().detach().numpy()[0]
 
    def select_action_batch(self, states, deterministic=False):
        """
        Batch action selection สำหรับ multi-env training
        states: np.ndarray shape (N, state_dim)
        returns: np.ndarray shape (N, action_dim)
        """
        import numpy as np
        t = torch.FloatTensor(states).to(self.device)
 
        if deterministic:
            mean, _ = self.actor.forward(t)
            y       = torch.tanh(mean)
            actions = (self.actor.min_action
                       + (y + 1) * 0.5
                       * (self.actor.max_action - self.actor.min_action))
        else:
            actions, _ = self.actor.sample(t)
 
        # sync action_log กับ env[0] เพื่อให้ logger ไม่ได้รับ None
        self.action_log = actions[0].unsqueeze(0)
        return actions.cpu().detach().numpy()
 
 
    # ======================================================================
    # Training
    # ======================================================================
    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
 
        sample = self.replay_buffer.sample(batch_size)
 
        if len(sample) == 5:
            state, action, reward, next_state, done = sample
            indices, is_weights = None, None
        else:
            state, action, reward, next_state, done, indices, is_weights = sample
 
        # ------------------------------------------------------------
        # numpy / torch → torch (unified)
        # ------------------------------------------------------------
        def to_tensor(x):
            if torch.is_tensor(x):
                return x.to(self.device)
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)
 
        state      = to_tensor(state)
        action     = to_tensor(action)
        reward     = to_tensor(reward)
        next_state = to_tensor(next_state)
        done       = to_tensor(done)
 
        if is_weights is not None:
            is_weights = to_tensor(is_weights)
 
        # ------------------------------------------------------------
        # Target Q (⚠️ N-step aware)
        # ------------------------------------------------------------
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_state)
            q1_t, q2_t = self.target_critic(next_state, next_action)
            q_min = torch.min(q1_t, q2_t)
 
            # IMPORTANT:
            # reward จาก n-step buffer คือ R_n แล้ว → ไม่คูณ gamma ซ้ำ
            target_q = reward + (1.0 - done) * (
                q_min - self.alpha * next_logp
            )
 
            # numerical safety
            target_q = torch.clamp(target_q, -1e6, 1e6)
 
        # ------------------------------------------------------------
        # Critic update
        # ------------------------------------------------------------
        q1, q2 = self.critic(state, action)
 
        td_error1 = q1 - target_q
        td_error2 = q2 - target_q
        td_error  = 0.5 * (td_error1.abs() + td_error2.abs())
 
        if is_weights is not None:
            critic_loss = (
                is_weights * (td_error1.pow(2) + td_error2.pow(2))
            ).mean()
        else:
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
 
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
 
        # ------------------------------------------------------------
        # Update priorities (PER-safe)
        # ------------------------------------------------------------
        if indices is not None:
            safe_td = torch.clamp(td_error, 0.0, 1e6)
            self.replay_buffer.update_priorities(
                indices, safe_td.detach().squeeze(-1)
            )
 
        # ------------------------------------------------------------
        # Actor update (guard NaN)
        # ------------------------------------------------------------
        a_pi, log_pi = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
 
        if torch.isnan(q_pi).any() or torch.isnan(log_pi).any():
            return  # skip unstable step
 
        actor_loss = (self.alpha * log_pi - q_pi).mean()
 
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
 
        # ------------------------------------------------------------
        # Soft update target critic
        # ------------------------------------------------------------
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
 
        # ------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------
        if self.logger_status:
            self.logger.log("loss_actor",  actor_loss.item())
            self.logger.log("loss_critic", critic_loss.item())
            self.logger.log("q1_mean",     q1.mean().item())
            self.logger.log("q2_mean",     q2.mean().item())
            self.logger.log("entropy",     -log_pi.mean().item())
            self.logger.log("alpha",       self.alpha)
            self.logger.log("tau",         self.tau)
            self.logger.log("action",      self.action_log)
 
    # ======================================================================
    # Checkpoint: Save everything (model + optimizers + episode)
    # ======================================================================
    def save_checkpoint(self, episode, path="checkpoints/sac_checkpoint.pt"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
 
        checkpoint = {
            "episode": episode,
 
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
 
            "actor_opt":  self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
 
            "hyperparams": {
                "gamma": self.gamma,
                "tau":   self.tau,
                "alpha": self.alpha,
 
                # Action bounds
                "min_action": self.actor.min_action.cpu().tolist(),
                "max_action": self.actor.max_action.cpu().tolist(),
 
                # Actor architecture
                "state_dim":                  self.actor.state_dim,
                "action_dim":                 self.actor.action_dim,
                "simple_layers_actor":        self.simple_layers_actor,
                "simple_hidden_actor":        self.simple_hidden_actor,
                "advanced_hidden_size_actor": self.advanced_hidden_size_actor,
 
                # Critic architecture
                "simple_layers_critic":         self.simple_layers_critic,
                "simple_hidden_critic":         self.simple_hidden_critic,
                "advanced_hidden_sizes_critic": self.advanced_hidden_sizes_critic,
                "critic_encoder":               self.critic_encoder,
            }
        }
 
        torch.save(checkpoint, path)
        print(f"[AutoSave] Saved checkpoint at episode {episode}")
 
    # ======================================================================
    # Load Checkpoint
    # ======================================================================
    def load_checkpoint(self, path="checkpoints/sac_checkpoint.pt"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
 
        data  = torch.load(path, map_location=self.device)
        hyper = data["hyperparams"]
 
        self.gamma = hyper["gamma"]
        self.tau   = hyper["tau"]
        self.alpha = hyper["alpha"]
 
        state_dim  = hyper["state_dim"]
        action_dim = hyper["action_dim"]
 
        min_action = torch.tensor(hyper["min_action"], dtype=torch.float32)
        max_action = torch.tensor(hyper["max_action"], dtype=torch.float32)
 
        # Actor config
        self.simple_layers_actor        = hyper["simple_layers_actor"]
        self.simple_hidden_actor        = hyper["simple_hidden_actor"]
        self.advanced_hidden_size_actor = hyper["advanced_hidden_size_actor"]
 
        # Critic config
        self.simple_layers_critic         = hyper["simple_layers_critic"]
        self.simple_hidden_critic         = hyper["simple_hidden_critic"]
        self.advanced_hidden_sizes_critic = hyper["advanced_hidden_sizes_critic"]
        self.critic_encoder               = hyper["critic_encoder"]
 
        # ---- Rebuild Actor ----
        self.actor = Actor(
            state_dim, action_dim,
            min_action, max_action,
            simple_layers=self.simple_layers_actor,
            simple_hidden=self.simple_hidden_actor,
            advanced_hidden_sizes=self.advanced_hidden_size_actor
        ).to(self.device)
        self.actor.load_state_dict(data["actor"])
 
        # ---- Rebuild Critic ----
        self.critic = Critic(
            state_dim, action_dim,
            simple_layers=self.simple_layers_critic,
            simple_hidden=self.simple_hidden_critic,
            advanced_hidden_sizes=self.advanced_hidden_sizes_critic,
            use_encoder=self.critic_encoder
        ).to(self.device)
        self.critic.load_state_dict(data["critic"])
 
        self.target_critic = Critic(
            state_dim, action_dim,
            simple_layers=self.simple_layers_critic,
            simple_hidden=self.simple_hidden_critic,
            advanced_hidden_sizes=self.advanced_hidden_sizes_critic,
            use_encoder=self.critic_encoder
        ).to(self.device)
        self.target_critic.load_state_dict(data["target_critic"])
 
        # ---- Load Optimizers ----
        self.actor_opt.load_state_dict(data["actor_opt"])
        self.critic_opt.load_state_dict(data["critic_opt"])
 
        print(f"[Resume] Loaded checkpoint from episode {data['episode']}")
        return data["episode"]
 
    # ======================================================================
    # Save Model (weights only)
    # ======================================================================
    def save_model(self, path="sac_model.pt"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
 
        data = {
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
 
            "hyperparams": {
                "gamma": self.gamma,
                "tau":   self.tau,
                "alpha": self.alpha,
 
                "min_action": self.actor.min_action.cpu().tolist(),
                "max_action": self.actor.max_action.cpu().tolist(),
 
                "state_dim":  self.actor.state_dim,
                "action_dim": self.actor.action_dim,
 
                # actor
                "simple_layers_actor":        self.simple_layers_actor,
                "simple_hidden_actor":        self.simple_hidden_actor,
                "advanced_hidden_size_actor": self.advanced_hidden_size_actor,
 
                # critic
                "simple_layers_critic":         self.simple_layers_critic,
                "simple_hidden_critic":         self.simple_hidden_critic,
                "advanced_hidden_sizes_critic": self.advanced_hidden_sizes_critic,
                "critic_encoder":               self.critic_encoder,
            }
        }
 
        torch.save(data, path)
        print(f"[SaveModel] Model saved → {path}")
 
    # ======================================================================
    # Load Model (weights only)
    # ======================================================================
    def load_model(self, path="sac_model.pt"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
 
        data  = torch.load(path, map_location=self.device)
        hyper = data["hyperparams"]
 
        # load hyper
        self.gamma = hyper["gamma"]
        self.tau   = hyper["tau"]
        self.alpha = hyper["alpha"]
 
        state_dim  = hyper["state_dim"]
        action_dim = hyper["action_dim"]
 
        min_action = torch.tensor(hyper["min_action"])
        max_action = torch.tensor(hyper["max_action"])
 
        # actor architecture
        self.simple_layers_actor        = hyper["simple_layers_actor"]
        self.simple_hidden_actor        = hyper["simple_hidden_actor"]
        self.advanced_hidden_size_actor = hyper["advanced_hidden_size_actor"]
 
        # critic architecture
        self.simple_layers_critic         = hyper["simple_layers_critic"]
        self.simple_hidden_critic         = hyper["simple_hidden_critic"]
        self.advanced_hidden_sizes_critic = hyper["advanced_hidden_sizes_critic"]
        self.critic_encoder               = hyper["critic_encoder"]
 
        # recreate actor
        self.actor = Actor(
            state_dim, action_dim,
            min_action, max_action,
            simple_layers=self.simple_layers_actor,
            simple_hidden=self.simple_hidden_actor,
            advanced_hidden_sizes=self.advanced_hidden_size_actor
        ).to(self.device)
        self.actor.load_state_dict(data["actor"])
 
        # recreate critic
        self.critic = Critic(
            state_dim, action_dim,
            simple_layers=self.simple_layers_critic,
            simple_hidden=self.simple_hidden_critic,
            advanced_hidden_sizes=self.advanced_hidden_sizes_critic,
            use_encoder=self.critic_encoder
        ).to(self.device)
        self.critic.load_state_dict(data["critic"])
 
        # recreate target critic
        self.target_critic = Critic(
            state_dim, action_dim,
            simple_layers=self.simple_layers_critic,
            simple_hidden=self.simple_hidden_critic,
            advanced_hidden_sizes=self.advanced_hidden_sizes_critic,
            use_encoder=self.critic_encoder
        ).to(self.device)
        self.target_critic.load_state_dict(data["target_critic"])
 
        print(f"[LoadModel] Loaded from {path}")
 
    # ======================================================================
    # Logger utilities
    # ======================================================================
    def reset_logger(self):
        self.logger = MetricLogger(
            folder=self.logger_path,
            filename=self.file_name_log,
            auto_increment=True
        )
        print("[Logger] Reset: created a new empty logger.")
 
    def clear_logger(self, key=None):
        self.logger.clear(key)
        if key is None:
            print("[Logger] Cleared all metrics.")
        else:
            print(f"[Logger] Cleared metric '{key}'.")
 