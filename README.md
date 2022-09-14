# User study: AI to Human teaching

# Research objective

The purpose of this research is to find a medium to transfer knowledge from reinforcement learning (RL) agents to humans. This medium should enable the latter to perform comparably with explanations given by other humans.

## Asumptions

This research is done assuming that the following statements are true:

1.  Humans are able to transfer knowledge to others about any computable domain using either natural language or a whiteboard.
2.  Each tested medium can be used to represent the policy of any RL agents (An inefficient way is valid).
3.  Each tested medium can be understood by most humans with a short explanation of their functioning.

This research will not attempt to prove those statements as it will ony be useful if the research results are positive.

## Hypotheses

This research will evaluate the following hypotheses:

1.  Some mediums are better than others when it comes to transfering knowledge to humans.
2.  Providing [Hierarchical Behavior Explanations as Graphs](https://github.com/IRLL/options_graphs) (HBEG) is comparable to an other human explanation on the following evaluation metrics and domains.

## Evaluation Metrics

1.  Best performance on an episode of the task
2.  Real time speed of performance increase of the task
3.  Ability to reproduce the task without any given explanation after a short time

## Knowledge Mediums

-   Nothing (Human must learn by reinforcment itself)
-   Step instruction (Output of the policy)
-   Other human written instructions
-   Other human graphical instructions
-   [Hierarchical Behavior Explanations as Graphs](https://github.com/IRLL/options_graphs) (HBEG)

## Domains

-   The [Crafting environment](https://github.com/IRLL/Crafting) for arbitrary hierarchical tasks.
-   The [MineCrafting environment](https://github.com/IRLL/Crafting/tree/master/crafting/examples/minecraft) for MineCraft hierarchical tasks without the complex 3D navigation.
-   The [Minigrid environment](https://github.com/MathisFederico/gym-minigrid/) for classic 2D navigation.

# Installation

1.  git clone this repository.

2.  Initialize submodules:

```bach
git submodule update --init --recursive
```

3.  Install requirements

```bach
pip install -r requirements.txt
```

4.  Install domains of interest

Crafting

```bach
pip install -e .\crafting
```

Minigrid

```bach
pip install -e .\minigrid
```

# Quickstart

## Crafting

### Manual run

```bach
python -m crafting.examples.minecraft
```

<a href="https://github.com/MathisFederico/Crafting">
  <img src="./docs/gifs/MineCrafting.gif" alt="MineCrafting">
</a>

### HippoGym run

Enter the submodule:

```bach
cd hippo_gym
```

Install requirements:

```bach
pip install -r requirements.txt
```

Launch the local hosted server in dev mode:

```bach
python -m App dev
```

Go to local hosted frontend in a browser: [App](https://testing.irll.net/?server=ws://localhost:5000) or [Debug](https://irll.net/?server=ws://localhost:5000&debug=true).

<a href="https://testing.irll.net/?server=ws://localhost:5000">
  <img src="./docs/gifs/Crafting-Hippogym.gif" alt="MineCrafting on HippoGym">
</a>

## MiniGrid

### KeyDoor

```bach
python -m minigrid.manual_control --env MiniGrid-DoorKey-8x8-v0 --agent_view
```

<a href="https://github.com/maximecb/gym-minigrid">
  <img src="./docs/gifs/Minigrid-KeyDoor-Demo.gif" alt="Minigrid-KeyDoor">
</a>
