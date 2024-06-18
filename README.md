# Constrained Human-AI Cooperation (CHAIC): An Inclusive Embodied Social Intelligence Challenge

## ✨ Introduction
This is the anonymous raw code for the NeurIPS Dataset Track submission CHAIC.

You could view the [[Project Page](https://chaic-neurips.github.io/CHAIC/)] for some video demos.

> We introduce the Constrained Human-AI Cooperation (CHAIC), an inclusive embodied social intelligence challenge for testing social perception and cooperation in embodied agents. In CHAIC, the goal is for an embodied agent equipped with egocentric observations to aid a human possibly operating under physical constraints, e.g. unable to reach high places or confined to a wheelchair, to perform common household or outdoor tasks as efficiently as possible. To do this, a successful helper must (1). infer the human's intents and constraints by following the human and observing their behaviors (social perception), and (2). make a cooperative plan tailored to the human user to solve the task as fast as possible together as a team (cooperative planning). 
To benchmark this challenge, we created 4 new agents with real physical constraints, and 8 long-horizon tasks featuring both indoor and outdoor scenes with various constraints and emergency events along with potential risks. We benchmark both planning and learning baselines on the challenge and introduce a new method leveraging Large Language Models and behavior modeling. Empirical evaluation demonstrates the ability of our benchmark to enable systematic evaluation of important elements of machine social intelligence.

<div>
<center>
<img src="docs/figure/teaser_v4.png">
</div>

## 🛠️ Setup

**Step 1:** Run the following commands step by step to set the environments:

```bash
conda create -n CHAIC python=3.9
conda activate CHAIC
pip3 install -e .
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

If you are running in a remote server without a screen, please refer to [running TDW in server](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/server.md).

After that, you can run the demo scene to verify your setup:

```bash
python demo/demo_scene.py
```

**Step 2:** Install and download pre-trained perception models:

```bash
pip install -U openmim
mim install mmengine
mim install mmdet
pip install mmaction2
bash detection_pipeline/download_ckpt.sh
```

After that, you can run the perception demos to verify them:

```bash
python tdw-gym/detection.py
python tdw-gym/behavior.py
```

**Notice:** There may exist some internal bugs in the `mmaction` package, and you can refer to the [Github issue](https://github.com/open-mmlab/mmaction2/issues/2714) to fix it when you meet trouble.

## 💾 Codebase Layout
Some important folders and their corresponding functions are listed here.
```
|__ tdw-gym/                         Main code
|
|__ scenes/                          Code for dataset generation
|
|__ dataset/                         Dataset configuration and storage
|
|__ transport_challenge_multi_agent/ Low level controller
|
|__ scripts/                         Scripts for running experiments
|
|__ detection_pipeline/              Code for perception models
|
|__ LM_agent/                        LLM & VLM Prompt
```

## 💫 Run Experiments

We prepare all the experiment scripts to run experiments under the folder `scripts`. For example, to run experiments with Random Helper in the highthing setting, you can use the following command:

```bash
bash scripts/random_helper/test_high_thing_random_helper.sh
```

By adding ``--gt_mask`` or ``--gt_behavior`` in the scripts, the environment will provide ground truth object segmentation masks or ground truth behaviors of the partner, respectively.

**Notice:** If you want to test the LLM+BM helper or the VLM helper, you need to fill your ``AzureOpenAI`` setting or ``OPENAI_API_KEY`` at lines 73-77 in ``LM_agent/LLM.py`` or lines 74-78 in ``LM_agent/VLM.py``.

## 🧾 Benchmark Highlights

### Multi-Agent Asynchronized Setting

Agents may take different number of frames to finish (or fail) one action, and one env step is finished until any agent's action is not under the ongoing status, and the current obs is returned to all agents. Then, 
all agents are asked for a new action, and any agent having ongoing action will directly switch to the new action if its action changes. 

### Heterogeneity of Agents

Different types of agents have different capacity scopes, and agents with different capacity scopes need to work together to achieve a common goal. Meanwhile, although the task goal is the same for all agents, the constrained agent and the helper will receive different information about the goal: The constrained agent can know the exact goal of the task, while the helper needs to perceive the constrained agent's behavior and infer the true goal.

### Realistic Observation

One goal of our benchmark is to mimic real life as similar as possible. Therefore, we only provide the raw RGBD images as the main observation (the benchmark also supports many other types of observation), making our benchmark challenging and having a wide range of application space.

## 🏆 Results

The table below is the quantitative results on CHAIC benchmark. We report the average Transport Rate (TR), Efficiency Improvement (EI), Goal Inference Accuracy (IA), Completion Ratio of Helper (CR) and Standard Error of Transport Rate (STD_TR) here. w/o means the main agent does the task solely without a helper. The Emergency Rate (ER) metric is also reported for the shopping task. 

<div class="center">
    <table>
        <tr>
            <td colspan="1"> <b>TR(EI)<span>&#8593;</span></b> </td>
            <td colspan="6" style="text-align: center">Indoor</td>
            <td colspan="2" style="text-align: center">Outdoor</td>
        </tr>
        <tr>
            <td>Helper Agent</td>
            <td>Normal</td>
            <td>High Target</td>
            <td>High Container</td>
            <td>High Goalplace</td>
            <td>Lowthing</td>
            <td>Wheelchair</td>
            <td>Shopping</td>
            <td>Furniture</td>
        </tr>
        <tr>
            <td>w/o</td>
            <td>0.53</td>
            <td>0.30</td>
            <td>0.38</td>
            <td>0.27</td>
            <td>0.51</td>
            <td>0.08</td>
            <td>0.37</td>
            <td>0.17</td>
        </tr>
        <tr>
            <td>Random</td>
            <td>0.52(-0.02)</td>
            <td>0.27(-0.05)</td>
            <td>0.34(-0.05)</td>
            <td>0.33(0.10)</td>
            <td>0.51(0.00)</td>
            <td>0.21(0.86)</td>
            <td>0.39(0.05)</td>
            <td>0.48(0.67)</td>
        </tr>
        <tr>
            <td>RHP</td>
            <td>0.63(0.15)</td>
            <td>0.35(0.10)</td>
            <td>0.44(0.18)</td>
            <td><b>0.35</b>(0.17)</td>
            <td>0.66(0.23)</td>
            <td><b>0.44</b>(0.77)</td>
            <td>0.49(0.22)</td>
            <td>0.64(0.72)</td>
        </tr>
        <tr>
            <td>RL</td>
            <td>0.45(-0.19)</td>
            <td>0.26(-0.16)</td>
            <td>0.28(-0.25)</td>
            <td>0.25(-0.22)</td>
            <td>0.43(-0.16)</td>
            <td>0.11(0.07)</td>
            <td>0.32(-0.13)</td>
            <td>0.67(0.74)</td>
        </tr>
        <tr>
            <td>SmartHelp</td>
            <td>0.46(-0.12)</td>
            <td>0.24(-0.17)</td>
            <td>0.26(-0.28)</td>
            <td>0.31(0.01)</td>
            <td>0.49(-0.04)</td>
            <td>0.13(0.11)</td>
            <td>0.32(-0.13)</td>
            <td>0.57(0.70)</td>
        </tr>
        <tr>
            <td>VLM</td>
            <td>0.65(0.16)</td>
            <td>0.32(0.04)</td>
            <td>0.37(-0.01)</td>
            <td>0.20(-0.46)</td>
            <td>0.68(0.24)</td>
            <td><b>0.44(0.86)</b></td>
            <td>0.49(0.20)</td>
            <td><b>0.69(0.76)</b></td>
        </tr>
        <tr style="border-bottom: 3px solid black">
            <td>LLM+BM</td>
            <td><b>0.66(0.18)</b></td>
            <td><b>0.38(0.11)</b></td>
            <td><b>0.46(0.17)</b></td>
            <td><b>0.35(0.18)</b></td>
            <td><b>0.70(0.26)</b></td>
            <td>0.40(0.80)</td>
            <td><b>0.60(0.36)</b></td>
            <td>0.67(0.73)</td>
        </tr>
        <tr>
            <td>Oracle</td>
            <td>0.78(0.32)</td>
            <td>0.49(0.28)</td>
            <td>0.72(0.48)</td>
            <td>0.62(0.55)</td>
            <td>0.84(0.39)</td>
            <td>0.59(0.88)</td>
            <td>0.62(0.40)</td>
            <td>0.70(0.77)</td>
        </tr>
    </table>
    <table>
        <tr>
            <td colspan="1"> <b>IA<span>&#8593;</span></b> </td>
            <td colspan="6" style="text-align: center">Indoor</td>
            <td colspan="1" style="text-align: center">Outdoor</td>
        </tr>
        <tr>
            <td>Helper Agent</td>
            <td>Normal</td>
            <td>High Target</td>
            <td>High Container</td>
            <td>High Goalplace</td>
            <td>Lowthing</td>
            <td>Wheelchair</td>
            <td>Shopping</td>
        </tr>
        <tr>
            <td>Random</td>
            <td>0.24</td>
            <td>0.29</td>
            <td>0.17</td>
            <td>0.14</td>
            <td>0.31</td>
            <td>0.24</td>
            <td>0.34</td>
        </tr>
        <tr>
            <td>RHP</td>
            <td>0.15</td>
            <td>0.29</td>
            <td>0.20</td>
            <td>0.21</td>
            <td>0.28</td>
            <td>0.17</td>
            <td>0.44</td>
        </tr>
        <tr>
            <td>VLM</td>
            <td>0.20</td>
            <td>0.00</td>
            <td>0.20</td>
            <td>0.20</td>
            <td><b>0.44</b></td>
            <td><b>0.59</b></td>
            <td>0.59</td>
        </tr>
        <tr style="border-bottom: 3px solid black">
            <td>LLM+BM</td>
            <td><b>0.27</b></td>
            <td><b>0.33</b></td>
            <td><b>0.28</b></td>
            <td><b>0.33</b></td>
            <td>0.40</td>
            <td>0.38</td>
            <td><b>0.70</b></td>
        </tr>
        <tr>
            <td>Oracle</td>
            <td>0.87</td>
            <td>0.91</td>
            <td>0.89</td>
            <td>0.90</td>
            <td>0.88</td>
            <td>0.82</td>
            <td>0.89</td>
        </tr>
    </table>
    <table>
        <tr>
            <td colspan="1"> <b>CR<span>&#8593;</span></b> </td>
            <td colspan="6" style="text-align: center">Indoor</td>
            <td colspan="2" style="text-align: center">Outdoor</td>
        </tr>
        <tr>
            <td>Helper Agent</td>
            <td>Normal</td>
            <td>High Target</td>
            <td>High Container</td>
            <td>High Goalplace</td>
            <td>Lowthing</td>
            <td>Wheelchair</td>
            <td>Shopping</td>
            <td>Furniture</td>
        </tr>
        <tr>
            <td>Random</td>
            <td>0.08</td>
            <td>0.10</td>
            <td>0.07</td>
            <td>0.06</td>
            <td>0.09</td>
            <td>0.09</td>
            <td>0.07</td>
            <td>0.73</td>
        </tr>
        <tr>
            <td>RHP</td>
            <td>0.15</td>
            <td><b>0.42</b></td>
            <td><b>0.26</b></td>
            <td><b>0.39</b></td>
            <td>0.35</td>
            <td>0.19</td>
            <td>0.33</td>
            <td>0.74</td>
        </tr>
        <tr>
            <td>VLM</td>
            <td>0.11</td>
            <td>0.00</td>
            <td>0.23</td>
            <td>0.05</td>
            <td><b>0.40</b></td>
            <td>0.15</td>
            <td>0.32</td>
            <td><b>0.81</b></td>
        </tr>
        <tr style="border-bottom: 3px solid black">
            <td>LLM+BM</td>
            <td><b>0.23</b></td>
            <td>0.32</td>
            <td>0.25</td>
            <td>0.38</td>
            <td>0.35</td>
            <td><b>0.43</b></td>
            <td><b>0.42</b></td>
            <td>0.73</td>
        </tr>
        <tr>
            <td>Oracle</td>
            <td>0.50</td>
            <td>0.67</td>
            <td>0.66</td>
            <td>0.69</td>
            <td>0.55</td>
            <td>0.36</td>
            <td>0.46</td>
            <td>0.86</td>
        </tr>
    </table>
    <table>
        <tr>
            <td colspan="1"> <b>STD</b></td>
            <td colspan="6" style="text-align: center">Indoor</td>
            <td colspan="2" style="text-align: center">Outdoor</td>
        </tr>
        <tr>
            <td>Helper Agent</td>
            <td>Normal</td>
            <td>High Target</td>
            <td>High Container</td>
            <td>High Goalplace</td>
            <td>Lowthing</td>
            <td>Wheelchair</td>
            <td>Shopping</td>
            <td>Furniture</td>
        </tr>
        <tr>
            <td>w/o</td>
            <td>0.17</td>
            <td>0.10</td>
            <td>0.07</td>
            <td>0.06</td>
            <td>0.18</td>
            <td>0.20</td>
            <td>0.11</td>
            <td>0.18</td>
        </tr>
        <tr>
            <td>Random</td>
            <td>0.19</td>
            <td>0.14</td>
            <td>0.16</td>
            <td>0.20</td>
            <td>0.19</td>
            <td>0.21</td>
            <td>0.11</td>
            <td>0.27</td>
        </tr>
        <tr>
            <td>RHP</td>
            <td>0.11</td>
            <td>0.18</td>
            <td>0.15</td>
            <td>0.22</td>
            <td>0.13</td>
            <td>0.19</td>
            <td>0.15</td>
            <td>0.22</td>
        </tr>
        <tr>
            <td>VLM</td>
            <td>0.16</td>
            <td>0.09</td>
            <td>0.12</td>
            <td>0.24</td>
            <td>0.12</td>
            <td>0.13</td>
            <td>0.15</td>
            <td>0.25</td>
        </tr>
        <tr style="border-bottom: 3px solid black">
            <td>LLM+BM</td>
            <td>0.14</td>
            <td>0.16</td>
            <td>0.18</td>
            <td>0.15</td>
            <td>0.09</td>
            <td>0.23</td>
            <td>0.19</td>
            <td>0.28</td>
        </tr>
        <tr>
            <td>Oracle</td>
            <td>0.15</td>
            <td>0.18</td>
            <td>0.10</td>
            <td>0.21</td>
            <td>0.14</td>
            <td>0.19</td>
            <td>0.13</td>
            <td>0.23</td>
        </tr>
    </table>
    <table>
        <tr>
            <td colspan="1"> <b>ER<span>&#8595;</span></b> </td>
            <td colspan="1" style="text-align: center">Outdoor</td>
        </tr>
        <tr>
            <td>Helper Agent</td>
            <td>Shopping</td>
        </tr>
        <tr>
            <td>Random</td>
            <td>956.7</td>
        </tr>
        <tr>
            <td>RHP</td>
            <td><b>887.5</b></td>
        </tr>
        <tr>
            <td>VLM</td>
            <td>2030.7</td>
        </tr>
        <tr style="border-bottom: 3px solid black">
            <td>LLM+BM</td>
            <td>1288.8</td>
        </tr>
        <tr>
            <td>Oracle</td>
            <td>445.6</td>
        </tr>
    </table>
</div>
