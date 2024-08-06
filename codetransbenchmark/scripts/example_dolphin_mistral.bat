@REM python ./scripts/run_commands.py translate dolphin-2.6-mistral via_description -a 1 -over

@REM python ./scripts/run_commands.py repair dolphin-2.6-mistral controlled_md -a 1 -over
@REM python ./scripts/run_commands.py clean_generations dolphin-2.6-mistral controlled_md -a 2 -over
@REM python ./scripts/run_commands.py test dolphin-2.6-mistral controlled_md -a 2 -over -d avatar
@REM python ./scripts/run_commands.py test dolphin-2.6-mistral controlled_md -a 2 -over -d bithacks

@REM python ./scripts/run_commands.py repair dolphin-2.6-mistral controlled -a 1 -over
@REM python ./scripts/run_commands.py clean_generations dolphin-2.6-mistral controlled -a 2 -over
@REM python ./scripts/run_commands.py test dolphin-2.6-mistral controlled -a 2 -over

@REM python ./scripts/run_commands.py repair dolphin-2.6-mistral controlled -a 1 -over -d bithacks
@REM python ./scripts/run_commands.py clean_generations dolphin-2.6-mistral controlled -a 2
@REM python ./scripts/run_commands.py test dolphin-2.6-mistral controlled -a 2 -over -d bithacks

@REM python ./scripts/run_commands.py repair dolphin-2.6-mistral controlled_md -a 2 -over
@REM python ./scripts/run_commands.py repair dolphin-2.6-mistral controlled -a 2 -over


@REM python ./scripts/run_commands.py test dolphin-2.6-mistral controlled_md -a 1 -over -d avatar
@REM python ./scripts/run_commands.py repair dolphin-2.6-mistral controlled_md -a 1 -over -d avatar
@REM python ./scripts/run_commands.py clean_generations dolphin-2.6-mistral controlled_md -a 2
@REM python ./scripts/run_commands.py test dolphin-2.6-mistral controlled_md -a 2 -over -d avatar

python ./scripts/run_commands.py repair dolphin-2.6-mistral controlled_md -a 2 -over -d avatar
python ./scripts/run_commands.py clean_generations dolphin-2.6-mistral controlled_md -a 3 -over -d avatar
python ./scripts/run_commands.py test dolphin-2.6-mistral controlled_md -a 3 -over -d avatar
