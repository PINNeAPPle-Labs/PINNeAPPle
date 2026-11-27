- **`pinneapple plugins list | inspect | doctor | enable | disable`** (E3/#1474): five checks per plugin in the
  contract's order — the entry point imports, the contract version is compatible, the declared dependencies exist
  (including external executables, with the path found), the required configuration keys are present, `initialize()`
  succeeds. `doctor` tells "not installed" from "installed but failing" from "disabled"; `enable`/`disable` write
  `~/.pinneapple/plugins.json` (`$PINNEAPPLE_PLUGINS_CONFIG`) and never touch site-packages.
