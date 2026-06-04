1. Morris SSM Spatial Memory Agent

A simple Morris Water Maze simulation with:

- a mouse-like agent;
- persistent spatial memory saved in `outputs/`;
- visual cues around the tank;
- a self-supervised State Space Model (SSM);
- no reinforcement learning.

The agent learns from its own navigation sequence:

# current observation + action + internal hidden state -> next observation prediction

The prediction error is used only as a self-supervised learning signal.
There are no rewards, no Q-table, no DQN, no policy-gradient algorithm.

2. Why State Space Models?

The SSM keeps an internal hidden state. This hidden state works like a compact temporal memory of the recent trajectory:

h_t = f(h_{t-1}, observation_t, action_t)
prediction_t = g(h_t)


The agent uses:

1. the persistent spatial map;
2. the remembered platform position;
3. the best visual cue associated with the platform;
4. the SSM hidden state to make movement less random over time.



3. Run


> pip install -r requirements.txt
> python main.py


4. Controls

- M: show/hide mental map
- N: new trial, keep memory
- C: clear memory and model
- ESC: quit

5. Outputs


outputs/mental_map.json
outputs/mental_map.csv
outputs/mental_map.png
outputs/trial_metrics.csv
outputs/biological_interpretation.txt
outputs/ssm_model.pt
outputs/ssm_training_log.csv

