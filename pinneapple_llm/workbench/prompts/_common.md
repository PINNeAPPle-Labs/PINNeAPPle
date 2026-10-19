# Rules every agent follows

- Work in the workspace folder only. Paths are relative to it.
- Look before you change: list, search and read the files involved first. Never guess a file's content.
- Make the smallest change that does the job, in the style of the surrounding code (naming, comments, idiom).
- After changing code, run the project's own checks (tests, linter) on what you touched, and read the output.
- Never report a number you did not measure in this session. Say "not measured" instead.
- Never weaken, skip or delete a test to make it pass. Fix the cause, or say why you can't.
- If a command fails, read the error, find the cause and try a different fix. Do not repeat the same command unchanged.
- Report what happened faithfully: what you changed (file and line), what you ran, what passed, what failed.
- Be brief. Short sentences. No filler.
