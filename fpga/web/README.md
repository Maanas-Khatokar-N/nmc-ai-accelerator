# NMC web console

Upload an image in the browser, press **Run on accelerator**, and the board does the rest.

## 1. Try the interface (no board needed)
    python3 server.py        # config.json ships with "demo": true
    open http://localhost:8080

## 2. Put it on the board
Copy the `nmc-web` folder to the board (Jupyter file browser > Upload, or scp), then in the Jupyter terminal:

    cd nmc-web
    nano config.json         # fill in workdir, bringup_cmd, run_cmd; set "demo": false
    sudo python3 server.py   # (Jupyter terminals are usually root already; then plain python3 is fine)

Open `http://<board-ip>:8080` from any laptop or phone on the same network.

### config.json essentials
- `workdir`     folder you `cd` into before the commands
- `bringup_cmd` the bring-up command, runs once per server start (or via "Re-initialise board")
- `run_cmd`     the runnetwork command; `{image}` is replaced by the uploaded image path
- `highlights`  regexes that turn lines of your output into the big metric tiles
- `result_json` optional: if your run script writes
  `{"predictions":[{"label":"tabby cat","score":61.4}],"metrics":{"GOPS":96},"details":{...}}`
  to this file, it is used directly (most robust option)

Everything printed by your commands also appears in the Console panel, and every
`key: value` / `key=value` line shows up under "Show full performance details".
