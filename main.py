from __future__ import annotations

NUM_BANDITS = 10  # Only change this if you need something other than 10 or 3

import importlib
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import bandits_env
import algo as user_algo

APP_TITLE = "Bandits Testbed (Manual vs Algorithm)"
BUTTON_SIZE = (8, 3)

STANDARD_DICE = (4, 6, 8, 10, 12, 20)


def _expected_value_for_arm(env: bandits_env.BanditEnvironment, idx: int) -> float:
    """Return the true expected reward of an arm for metrics."""
    if env.kind == "gaussian":
        return float(env.gaussian.means[idx])
    else:
        n = int(env.dice.sides[idx])
        return (n + 1) / 2.0


def _best_arm_and_value(env: bandits_env.BanditEnvironment) -> tuple[int, float]:
    vals = np.array([_expected_value_for_arm(env, i) for i in range(env.num_arms)], dtype=float)
    j = int(np.argmax(vals))
    return j, float(vals[j])


class BanditsApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)

        # Randomness
        self.rng = np.random.default_rng()

        # Environment
        self.num_arms = NUM_BANDITS
        self.env = bandits_env.BanditEnvironment(self.num_arms, rng=self.rng)
        # Default setting
        self.env.set_kind("gaussian")
        self.gauss_limits = {
            "mean_min": -1.0,
            "mean_max": 1.0,
            "var_min": 0.5,
            "var_max": 2.0,
        }
        self.env.reset_gaussian_random(**self.gauss_limits)

        self.t = 0  # global time step
        self.history: list[tuple[int, int, float]] = []
        self.action_counts = np.zeros(self.num_arms, dtype=int)
        self.q_estimates = np.zeros(self.num_arms, dtype=float)  # sample means

        # Auto-run controls
        self.mode_var = tk.StringVar(value="Manual")
        self.speed_ms = tk.IntVar(value=300)
        self.total_steps_var = tk.IntVar(value=200)
        self.running = False

        # Compare state
        self.compare_running = False
        self.compare_speed_ms = tk.IntVar(value=200)
        self.compare_steps_var = tk.IntVar(value=200)
        self.compare_seed_var = tk.IntVar(value=42)
        self.compare_freeze_env = tk.BooleanVar(value=True)
        self.algo1_name_var = tk.StringVar(value="algo1")
        self.algo2_name_var = tk.StringVar(value="algo2")
        self.algo1_mod = None
        self.algo2_mod = None

        # Per-algorithmic compare structures
        self.cmp = {
            "A": {},
            "B": {},
        }

        # UI
        self._init_ui()
        self._refresh_stats()
        self._update_metrics_block()  

    # UI setup ----------
    def _init_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Bandits Tab
        self.bandits_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.bandits_tab, text="Bandits")

        # Top controls
        bandits_controls = ttk.Frame(self.bandits_tab)
        bandits_controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.arm_range_label = ttk.Label(bandits_controls, text=self._arms_label_text())
        self.arm_range_label.pack(side=tk.LEFT)

        # Buttons
        self.buttons_frame = ttk.Frame(self.bandits_tab)
        self.buttons_frame.pack(side=tk.TOP, pady=6)

        self.arm_buttons: list[tk.Button] = []
        self._rebuild_buttons()

        # Stats
        self.stats_var = tk.StringVar(value="")
        stats_label = ttk.Label(self.bandits_tab, textvariable=self.stats_var, font=("TkDefaultFont", 10))
        stats_label.pack(side=tk.TOP, pady=(6, 2))

        # True params
        reveal_frame = ttk.Frame(self.bandits_tab)
        reveal_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 8))
        ttk.Button(reveal_frame, text="Reveal/Hide True Params", command=self._toggle_reveal).pack(side=tk.LEFT)
        self.reveal_shown = False
        self.reveal_text = tk.Text(self.bandits_tab, height=6, width=80)
        self.reveal_text.configure(state=tk.DISABLED)

        # Performance Tab ---
        self.perf_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.perf_tab, text="Performance & Controls")

        controls = ttk.LabelFrame(self.perf_tab, text="Run Controls")
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        # Mode
        ttk.Label(controls, text="Mode:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        mode_combo = ttk.Combobox(controls, textvariable=self.mode_var, values=["Manual", "Algorithm"], state="readonly", width=12)
        mode_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        # Speed slider
        ttk.Label(controls, text="Speed (ms/step):").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        speed = ttk.Scale(controls, from_=50, to=1000, orient=tk.HORIZONTAL, variable=self.speed_ms)
        speed.grid(row=0, column=3, sticky="we", padx=4, pady=4)
        controls.grid_columnconfigure(3, weight=1)

        # Total steps
        ttk.Label(controls, text="Total Steps:").grid(row=0, column=4, sticky="w", padx=4, pady=4)
        steps_entry = ttk.Entry(controls, textvariable=self.total_steps_var, width=8)
        steps_entry.grid(row=0, column=5, sticky="w", padx=4, pady=4)

        # Run controls
        start_btn = ttk.Button(controls, text="Start", command=self.start)
        stop_btn  = ttk.Button(controls, text="Stop", command=self.stop)
        reset_btn = ttk.Button(controls, text="Reset", command=self.reset_all)
        reload_btn= ttk.Button(controls, text="Reload Algorithm", command=self.reload_algo)
        start_btn.grid(row=1, column=0, padx=4, pady=6, sticky="we")
        stop_btn.grid( row=1, column=1, padx=4, pady=6, sticky="we")
        reset_btn.grid(row=1, column=2, padx=4, pady=6, sticky="we")
        reload_btn.grid(row=1, column=3, padx=4, pady=6, sticky="we")

        # Perf chart
        fig_frame = ttk.LabelFrame(self.perf_tab, text="Average Reward vs Time")
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Step t")
        self.ax.set_ylabel("Average Reward")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.figure, master=fig_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Basic Metrics under the plot ---
        metrics_frame = ttk.LabelFrame(self.perf_tab, text="Metrics")
        metrics_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        self.metrics_var = tk.StringVar(value="")
        ttk.Label(metrics_frame, textvariable=self.metrics_var, justify="left").pack(side=tk.LEFT, padx=6, pady=6)

        # Environment Controls ----
        env_box = ttk.LabelFrame(self.perf_tab, text="Environment")
        env_box.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        # Kind selector
        ttk.Label(env_box, text="Kind:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.env_kind_var = tk.StringVar(value="gaussian")
        kind_combo = ttk.Combobox(env_box, textvariable=self.env_kind_var, values=["gaussian", "dice"], state="readonly", width=10)
        kind_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        kind_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_env_kind())

        # Quick switch
        ttk.Button(env_box, text="Switch to Dice (3 arms)", command=lambda: self._switch_kind_and_arms("dice", 3)).grid(row=0, column=2, padx=4, pady=4)
        ttk.Button(env_box, text="Switch to Gaussian (10 arms)", command=lambda: self._switch_kind_and_arms("gaussian", 10)).grid(row=0, column=3, padx=4, pady=4)

        # Gaussian limit
        gauss_box = ttk.LabelFrame(env_box, text="Gaussian Limits (randomized per reset)")
        gauss_box.grid(row=1, column=0, columnspan=4, sticky="we", padx=4, pady=6)
        for c in range(4):
            gauss_box.grid_columnconfigure(c, weight=1)

        self.mean_min_var = tk.DoubleVar(value=self.gauss_limits["mean_min"])
        self.mean_max_var = tk.DoubleVar(value=self.gauss_limits["mean_max"])
        self.var_min_var  = tk.DoubleVar(value=self.gauss_limits["var_min"])
        self.var_max_var  = tk.DoubleVar(value=self.gauss_limits["var_max"])

        ttk.Label(gauss_box, text="mean_min").grid(row=0, column=0, sticky="w")
        ttk.Entry(gauss_box, textvariable=self.mean_min_var, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(gauss_box, text="mean_max").grid(row=0, column=2, sticky="w")
        ttk.Entry(gauss_box, textvariable=self.mean_max_var, width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(gauss_box, text="var_min").grid(row=1, column=0, sticky="w")
        ttk.Entry(gauss_box, textvariable=self.var_min_var, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(gauss_box, text="var_max").grid(row=1, column=2, sticky="w")
        ttk.Entry(gauss_box, textvariable=self.var_max_var, width=8).grid(row=1, column=3, sticky="w")

        ttk.Button(gauss_box, text="Re-randomize Gaussians", command=self._rerandomize_gaussians).grid(row=2, column=0, columnspan=4, sticky="we", pady=4)

        # Dice sizes
        dice_box = ttk.LabelFrame(env_box, text="Dice Sizes (per arm)")
        dice_box.grid(row=2, column=0, columnspan=4, sticky="we", padx=4, pady=6)
        dice_box.grid_columnconfigure(0, weight=1)

        self.dice_vars: list[tk.StringVar] = []
        self.dice_combos: list[ttk.Combobox] = []
        self._build_dice_controls(dice_box)

        # Compare Tab -----------
        self.compare_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.compare_tab, text="Compare (algo1 vs algo2)")

        top = ttk.LabelFrame(self.compare_tab, text="Compare Controls")
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text="Algo A:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        algo1_combo = ttk.Combobox(top, textvariable=self.algo1_name_var, values=["algo1"], state="readonly", width=12)
        algo1_combo.grid(row=0, column=1, padx=4, pady=4, sticky="w")

        ttk.Label(top, text="Algo B:").grid(row=0, column=2, padx=4, pady=4, sticky="w")
        algo2_combo = ttk.Combobox(top, textvariable=self.algo2_name_var, values=["algo2"], state="readonly", width=12)
        algo2_combo.grid(row=0, column=3, padx=4, pady=4, sticky="w")

        ttk.Label(top, text="Seed:").grid(row=0, column=4, padx=4, pady=4, sticky="w")
        ttk.Entry(top, textvariable=self.compare_seed_var, width=8).grid(row=0, column=5, padx=4, pady=4, sticky="w")

        self.compare_freeze_env_cb = ttk.Checkbutton(top, text="Freeze same environment", variable=self.compare_freeze_env)
        self.compare_freeze_env_cb.grid(row=0, column=6, padx=6, pady=4, sticky="w")

        ttk.Label(top, text="Speed (ms/step):").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Scale(top, from_=50, to=1000, orient=tk.HORIZONTAL, variable=self.compare_speed_ms).grid(row=1, column=1, padx=4, pady=4, sticky="we")
        top.grid_columnconfigure(1, weight=1)

        ttk.Label(top, text="Total Steps:").grid(row=1, column=2, padx=4, pady=4, sticky="w")
        ttk.Entry(top, textvariable=self.compare_steps_var, width=8).grid(row=1, column=3, padx=4, pady=4, sticky="w")

        ttk.Button(top, text="Start", command=self.compare_start).grid(row=1, column=4, padx=4, pady=4, sticky="we")
        ttk.Button(top, text="Stop", command=self.compare_stop).grid(row=1, column=5, padx=4, pady=4, sticky="we")
        ttk.Button(top, text="Reset", command=self.compare_reset).grid(row=1, column=6, padx=4, pady=4, sticky="we")

        # Compare plot
        cmp_fig_frame = ttk.LabelFrame(self.compare_tab, text="Average Reward vs Time (A vs B)")
        cmp_fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.cmp_figure = Figure(figsize=(6, 3.8), dpi=100)
        self.cmp_ax = self.cmp_figure.add_subplot(111)
        self.cmp_ax.set_xlabel("Step t")
        self.cmp_ax.set_ylabel("Average Reward")
        self.cmp_ax.grid(True, alpha=0.3)
        self.cmp_canvas = FigureCanvasTkAgg(self.cmp_figure, master=cmp_fig_frame)
        self.cmp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Compare metrics
        cmp_metrics = ttk.LabelFrame(self.compare_tab, text="Metrics")
        cmp_metrics.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        self.cmp_metrics_A = tk.StringVar(value="A: -")
        self.cmp_metrics_B = tk.StringVar(value="B: -")
        ttk.Label(cmp_metrics, textvariable=self.cmp_metrics_A, justify="left").pack(side=tk.LEFT, padx=8, pady=6)
        ttk.Label(cmp_metrics, textvariable=self.cmp_metrics_B, justify="left").pack(side=tk.LEFT, padx=8, pady=6)

    # Dice controls -----
    def _build_dice_controls(self, parent: ttk.LabelFrame):
        # Clear prior controls if any
        for child in list(parent.children.values()):
            child.destroy()

        ttk.Label(parent, text="Choose die for each arm (d4/d6/d8/d10/d12/d20)").grid(row=0, column=0, sticky="w", padx=4, pady=2)

        self.dice_vars = []
        self.dice_combos = []
        row = 1
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="w")

        dice_strings = [str(x) for x in STANDARD_DICE]
        for i in range(self.num_arms):
            var = tk.StringVar(value="6")
            combo = ttk.Combobox(frame, values=dice_strings, textvariable=var, width=4, state="readonly")
            ttk.Label(frame, text=f"{chr(ord('A')+i)}:").grid(row=0, column=2*i, padx=(4,0))
            combo.grid(row=0, column=2*i+1, padx=(0,8))
            self.dice_vars.append(var)
            self.dice_combos.append(combo)

        ttk.Button(parent, text="Apply Dice Sizes", command=self._apply_dice_sizes).grid(row=2, column=0, sticky="w", padx=4, pady=4)

    # Logic ----------
    def _manual_pull(self, idx: int):
        if self.mode_var.get() != "Manual":
            messagebox.showinfo("Mode", "Switch to Manual mode to click arms yourself.")
            return
        self._visual_press(idx)
        self._apply_pull(idx)

    def _visual_press(self, idx: int):
        btn = self.arm_buttons[idx]
        btn.config(relief=tk.SUNKEN, state=tk.ACTIVE)
        self.root.update_idletasks()

    def _set_button_label(self, idx: int, label: str):
        btn = self.arm_buttons[idx]
        btn.config(text=label, relief=tk.RAISED, state=tk.NORMAL)

    def _apply_pull(self, idx: int):
        reward = self.env.pull(idx)
        # Update counts and sample-mean estimate
        self.action_counts[idx] += 1
        n = self.action_counts[idx]
        old_q = self.q_estimates[idx]
        self.q_estimates[idx] = old_q + (reward - old_q) / n

        # Show reward on the button
        shown = f"{chr(ord('A')+idx)}\n{reward:+.2f}"
        self._set_button_label(idx, shown)

        # Log history, advance time, update stats
        self.history.append((self.t, idx, reward))
        self.t += 1
        self._refresh_stats()
        self._update_plot()
        self._update_metrics_block()

        if self.reveal_shown:
            self._update_reveal_panel()

    def _refresh_stats(self):
        total_r = sum(r for _, _, r in self.history) if self.history else 0.0
        avg_r = total_r / len(self.history) if self.history else 0.0
        counts_str = ", ".join(f"{chr(ord('A')+i)}:{c}" for i, c in enumerate(self.action_counts))
        self.stats_var.set(f"t={self.t} | total reward={total_r:.2f} | avg reward={avg_r:.3f} | counts [{counts_str}]")

    def _update_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Step t")
        self.ax.set_ylabel("Average Reward")
        self.ax.grid(True, alpha=0.3)

        if self.history:
            rewards = np.array([r for _, _, r in self.history], dtype=float)
            running_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
            self.ax.plot(np.arange(len(running_avg)), running_avg, linewidth=1.5)
        self.canvas.draw_idle()

    def _update_metrics_block(self):
        if not self.history:
            self.metrics_var.set("Cumulative: 0.00 | Avg: 0.000 | % Optimal: 0.0% | Regret: 0.00")
            return
        total_r = float(sum(r for _, _, r in self.history))
        avg_r = total_r / len(self.history)
        best_idx, best_val = _best_arm_and_value(self.env)
        pulls_best = self.action_counts[best_idx]
        pct_opt = (pulls_best / max(1, np.sum(self.action_counts))) * 100.0
        regret = best_val * len(self.history) - total_r
        self.metrics_var.set(
            f"Cumulative: {total_r:.2f} | Avg: {avg_r:.3f} | % Optimal: {pct_opt:.1f}% | Regret: {regret:.2f}"
        )

    def start(self):
        if self.mode_var.get() != "Algorithm":
            messagebox.showinfo("Mode", "Switch to Algorithm mode to auto-run.")
            return
        if self.running:
            return
        self.running = True
        try:
            user_algo.initialize(self.num_arms, self.rng)
        except Exception as e:
            self.running = False
            messagebox.showerror("Algorithm Error", f"initialize() failed:\n{e}")
            return
        self._schedule_step()

    def stop(self):
        self.running = False

    def _clear_learning_state(self):
        """Reset counters/history/UI labels without changing env parameters."""
        self.stop()
        self.t = 0
        self.history.clear()
        self.action_counts = np.zeros(self.num_arms, dtype=int)
        self.q_estimates = np.zeros(self.num_arms, dtype=float)
        for i, btn in enumerate(self.arm_buttons):
            btn.config(text=chr(ord('A') + i), relief=tk.RAISED, state=tk.NORMAL)
        self._refresh_stats()
        self._update_plot()
        self._update_metrics_block()
        if self.reveal_shown:
            self._update_reveal_panel()

    def reset_all(self):
        """Public reset: keep current env params; just clear learning state."""
        self._clear_learning_state()

    def reload_algo(self):
        try:
            importlib.reload(user_algo)
            messagebox.showinfo("Reload", "algo.py reloaded successfully.")
        except Exception as e:
            messagebox.showerror("Reload Error", f"Failed to reload algo.py:\n{e}")

    def _schedule_step(self):
        if not self.running:
            return
        if self.t >= max(0, int(self.total_steps_var.get())):
            self.stop()
            return
        delay = max(10, int(self.speed_ms.get()))
        self.root.after(delay, self._algorithm_step)

    def _algorithm_step(self):
        if not self.running:
            return
        try:
            a = user_algo.choose_action(
                t=self.t,
                num_actions=self.num_arms,
                q_estimates=self.q_estimates.copy(),
                action_counts=self.action_counts.copy(),
                history=self.history.copy(),
                rng=self.rng,
            )
        except Exception as e:
            self.stop()
            messagebox.showerror("Algorithm Error", f"choose_action() failed:\n{e}")
            return

        if not isinstance(a, int) or not (0 <= a < self.num_arms):
            self.stop()
            messagebox.showerror("Algorithm Error", f"choose_action() returned invalid arm: {a!r}")
            return

        self._visual_press(a)
        self._apply_pull(a)
        self._schedule_step()

    #  Helpers --------
    def _arms_label_text(self) -> str:
        return f"Arms: {self.num_arms} (A…{chr(ord('A')+self.num_arms-1)}) | Env: {self.env.kind}"

    def _rebuild_buttons(self):
        for child in list(self.buttons_frame.children.values()):
            child.destroy()
        self.arm_buttons = []
        for i in range(self.num_arms):
            label = chr(ord('A') + i)
            btn = tk.Button(
                self.buttons_frame,
                text=label,
                width=BUTTON_SIZE[0],
                height=BUTTON_SIZE[1],
                relief=tk.RAISED,
                command=lambda idx=i: self._manual_pull(idx),
            )
            btn.grid(row=0, column=i, padx=4, pady=4)
            self.arm_buttons.append(btn)

    def _set_num_arms(self, new_n: int):
        self.stop()
        self.num_arms = int(new_n)
        self.env.set_kind(self.env.kind, num_arms=self.num_arms)
        self.action_counts = np.zeros(self.num_arms, dtype=int)
        self.q_estimates = np.zeros(self.num_arms, dtype=float)
        self.history.clear()
        self.t = 0
        self._rebuild_buttons()
        self.arm_range_label.config(text=self._arms_label_text())
        self._refresh_stats()
        self._update_plot()
        self._update_metrics_block()

        # Rebuild dice controls
        for child in self.perf_tab.children.values():
            if isinstance(child, ttk.Labelframe) and child.cget("text") == "Environment":
                for gchild in child.children.values():
                    if isinstance(gchild, ttk.Labelframe) and gchild.cget("text").startswith("Dice Sizes"):
                        self._build_dice_controls(gchild)
                        break
                break

    def _apply_env_kind(self):
        kind = self.env_kind_var.get()
        if kind == "dice" and self.num_arms != 3:
            messagebox.showinfo("Dice Mode", "Dice mode uses exactly 3 arms. Switching arm count to 3.")
            self._switch_kind_and_arms("dice", 3)
            return
        self.env.set_kind(kind)
        self.reset_all()
        self.arm_range_label.config(text=self._arms_label_text())

    def _switch_kind_and_arms(self, kind: str, arms: int):
        self.env_kind_var.set(kind)
        self.env.set_kind(kind)
        self._set_num_arms(arms)
        if kind == "gaussian":
            self._rerandomize_gaussians()
        else:
            try:
                sizes = [int(v.get() or "6") for v in self.dice_vars]
                self.env.set_dice_sides(sizes)
            except Exception:
                self.env.set_dice_sides([6] * self.num_arms)
            self._clear_learning_state()
        if self.reveal_shown:
            self._update_reveal_panel()

    def _rerandomize_gaussians(self):
        # Read limits
        try:
            mean_min = float(self.mean_min_var.get())
            mean_max = float(self.mean_max_var.get())
            var_min  = float(self.var_min_var.get())
            var_max  = float(self.var_max_var.get())
        except Exception:
            messagebox.showerror("Input Error", "Gaussian limits must be numeric.")
            return
        self.gauss_limits = {
            "mean_min": mean_min,
            "mean_max": mean_max,
            "var_min": var_min,
            "var_max": var_max,
        }
        # Update env
        self.env.reset_gaussian_random(**self.gauss_limits)
        self._clear_learning_state()

    def _apply_dice_sizes(self):
        if self.env.kind != "dice":
            messagebox.showinfo("Dice Sizes", "Switch to Dice mode first.")
            return
        try:
            sizes = [int(v.get() or "6") for v in self.dice_vars]
            for n in sizes:
                if n not in STANDARD_DICE:
                    raise ValueError(f"Invalid die size {n}; allowed: {STANDARD_DICE}")
            self.env.set_dice_sides(sizes)
        except Exception as e:
            messagebox.showerror("Dice Error", str(e))
            return
        self._clear_learning_state()

    # Reveal panel -------
    def _toggle_reveal(self):
        if self.reveal_shown:
            self.reveal_text.pack_forget()
            self.reveal_shown = False
        else:
            self.reveal_text.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
            self.reveal_shown = True
            self._update_reveal_panel()

    def _update_reveal_panel(self):
        params = self.env.get_true_params()
        lines = []
        if params["kind"] == "gaussian":
            means = params["means"]
            vars_ = params["variances"]
            lines.append("TRUE PARAMETERS (Gaussian)")
            for i, (m, v) in enumerate(zip(means, vars_)):
                lines.append(f"  {chr(ord('A')+i)}: mean={m:+.4f}, variance={v:.4f}, std={np.sqrt(v):.4f}")
        else:
            sides = params["sides"]
            lines.append("TRUE PARAMETERS (Dice)")
            for i, n in enumerate(sides):
                lines.append(f"  {chr(ord('A')+i)}: d{int(n)} (uniform 1..{int(n)})")
        text = "\n".join(lines)
        self.reveal_text.configure(state=tk.NORMAL)
        self.reveal_text.delete("1.0", tk.END)
        self.reveal_text.insert("1.0", text)
        self.reveal_text.configure(state=tk.DISABLED)

    # Compare tab logic --------
    def _clone_env_params(self):
        """Return dict with current environment parameters to clone elsewhere."""
        p = self.env.get_true_params()
        return {"kind": self.env.kind, **p}

    def _build_env_from_params(self, params: dict, rng: np.random.Generator) -> bandits_env.BanditEnvironment:
        env = bandits_env.BanditEnvironment(self.num_arms, rng=rng)
        env.set_kind(params["kind"], num_arms=self.num_arms)
        if params["kind"] == "gaussian":
            env.gaussian.means = params["means"].copy()
            env.gaussian.variances = params["variances"].copy()
        else:
            env.dice.sides = params["sides"].copy()
        return env

    def compare_start(self):
        if self.compare_running:
            return
        # Load modules
        try:
            self.algo1_mod = importlib.import_module(self.algo1_name_var.get())
            self.algo2_mod = importlib.import_module(self.algo2_name_var.get())
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import algo modules:\n{e}")
            return

        # Seeded RNGs
        seed = int(self.compare_seed_var.get())
        rngA = np.random.default_rng(seed + 1)
        rngB = np.random.default_rng(seed + 2)

        # Build envs
        if self.compare_freeze_env.get():
            base_params = self._clone_env_params()
            envA = self._build_env_from_params(base_params, rngA)
            envB = self._build_env_from_params(base_params, rngB)
        else:
            envA = bandits_env.BanditEnvironment(self.num_arms, rng=rngA)
            envA.set_kind(self.env.kind, num_arms=self.num_arms)
            envB = bandits_env.BanditEnvironment(self.num_arms, rng=rngB)
            envB.set_kind(self.env.kind, num_arms=self.num_arms)
            if self.env.kind == "gaussian":
                envA.reset_gaussian_random(**self.gauss_limits)
                envB.reset_gaussian_random(**self.gauss_limits)
            else:
                # default dice to current dice settings
                envA.set_dice_sides(self.env.dice.sides)
                envB.set_dice_sides(self.env.dice.sides)

        # Init state holders
        self.cmp["A"] = {
            "env": envA,
            "t": 0,
            "history": [],
            "action_counts": np.zeros(self.num_arms, dtype=int),
            "q_estimates": np.zeros(self.num_arms, dtype=float),
            "rng": rngA,
            "mod": self.algo1_mod,
        }
        self.cmp["B"] = {
            "env": envB,
            "t": 0,
            "history": [],
            "action_counts": np.zeros(self.num_arms, dtype=int),
            "q_estimates": np.zeros(self.num_arms, dtype=float),
            "rng": rngB,
            "mod": self.algo2_mod,
        }

        # Initialize
        try:
            self.algo1_mod.initialize(self.num_arms, rngA)
            self.algo2_mod.initialize(self.num_arms, rngB)
        except Exception as e:
            messagebox.showerror("Algorithm Error", f"initialize() failed:\n{e}")
            return

        self.compare_running = True
        self._compare_schedule_step()

    def compare_stop(self):
        self.compare_running = False

    def compare_reset(self):
        self.compare_stop()
        self.cmp["A"].clear()
        self.cmp["B"].clear()
        self._update_compare_plot()
        self.cmp_metrics_A.set("A: -")
        self.cmp_metrics_B.set("B: -")

    def _compare_schedule_step(self):
        if not self.compare_running:
            return
        tA = self.cmp["A"]["t"]
        tB = self.cmp["B"]["t"]
        T = max(tA, tB)
        if T >= max(0, int(self.compare_steps_var.get())):
            self.compare_stop()
            return
        delay = max(10, int(self.compare_speed_ms.get()))
        self.root.after(delay, self._compare_step)

    def _compare_step(self):
        if not self.compare_running:
            return
        for key in ("A", "B"):
            self._compare_step_one(key)
        self._update_compare_plot()
        self._update_compare_metrics()
        self._compare_schedule_step()

    def _compare_step_one(self, key: str):
        S = self.cmp[key]
        env: bandits_env.BanditEnvironment = S["env"]
        t = S["t"]
        rng = S["rng"]
        mod = S["mod"]
        q = S["q_estimates"].copy()
        n = S["action_counts"].copy()
        hist = S["history"].copy()
        try:
            a = mod.choose_action(
                t=t,
                num_actions=self.num_arms,
                q_estimates=q,
                action_counts=n,
                history=hist,
                rng=rng,
            )
        except Exception as e:
            self.compare_stop()
            messagebox.showerror("Algorithm Error", f"{key}.choose_action() failed:\n{e}")
            return
        if not isinstance(a, int) or not (0 <= a < self.num_arms):
            self.compare_stop()
            messagebox.showerror("Algorithm Error", f"{key}.choose_action() returned invalid arm: {a!r}")
            return
        r = env.pull(a)
        S["action_counts"][a] += 1
        k = S["action_counts"][a]
        old = S["q_estimates"][a]
        S["q_estimates"][a] = old + (r - old) / k
        S["history"].append((t, a, float(r)))
        S["t"] = t + 1

    def _update_compare_plot(self):
        self.cmp_ax.clear()
        self.cmp_ax.set_xlabel("Step t")
        self.cmp_ax.set_ylabel("Average Reward")
        self.cmp_ax.grid(True, alpha=0.3)

        for key, label in (("A", "Algo A"), ("B", "Algo B")):
            S = self.cmp.get(key) or {}
            hist = S.get("history", [])
            if hist:
                rewards = np.array([r for _, _, r in hist], dtype=float)
                running_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
                self.cmp_ax.plot(np.arange(len(running_avg)), running_avg, linewidth=1.5, label=label)
        self.cmp_ax.legend(loc="best")
        self.cmp_canvas.draw_idle()

    def _update_compare_metrics(self):
        for key, var, name in (("A", self.cmp_metrics_A, "A"), ("B", self.cmp_metrics_B, "B")):
            S = self.cmp[key]
            env = S.get("env")
            hist = S.get("history", [])
            act = S.get("action_counts", np.zeros(self.num_arms, int))
            if not hist or env is None:
                var.set(f"{name}: -")
                continue
            total_r = float(sum(r for _, _, r in hist))
            avg_r = total_r / len(hist)
            best_idx, best_val = _best_arm_and_value(env)
            pct_opt = 100.0 * (act[best_idx] / max(1, np.sum(act)))
            regret = best_val * len(hist) - total_r
            var.set(f"{name} | Cumulative: {total_r:.2f} | Avg: {avg_r:.3f} | %Opt: {pct_opt:.1f}% | Regret: {regret:.2f}")


def main():
    root = tk.Tk()
    app = BanditsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
