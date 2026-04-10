# Triple-Star System Simulator

An interactive Streamlit app for exploring a Newtonian four-body system with three stars and one planet.

## What it does

- Simulates a 2D four-body gravitational system
- Uses a leapfrog (symplectic) integrator
- Visualizes trajectories, current positions, and the system barycenter
- Draws approximate osculating orbits
- Lets you change masses, initial positions, and simulation parameters

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Project structure

```text
.
├── app.py
├── LICENSE
├── README.md
├── requirements.txt
├── assets/
│   └── presets.json
├── physics/
│   ├── __init__.py
│   ├── init_conditions.py
│   ├── integrator.py
│   ├── osculating.py
│   ├── simulate.py
│   └── units.py
└── ui/
    ├── __init__.py
    ├── controls.py
    ├── draggable_canvas.py
    └── plotting.py
```

## Notes

- The app currently edits initial positions through numeric inputs, not drag-and-drop.
- `assets/presets.json` is included, but presets are not yet wired into the Streamlit UI.
- The simulation is a Newtonian approximation with no relativity, tides, or collision handling.
- Small input changes can produce very different outcomes because the system is often chaotic.

## Future improvements

- Add preset loading in the app
- Add a true drag-and-drop initial condition editor
- Add collision detection and merging
- Add 3D visualization
- Add energy and angular momentum diagnostics
- Swap in a higher-accuracy backend such as REBOUND/IAS15

## Contributing

Pull requests are welcome.
