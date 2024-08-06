@REM python ./scripts/run_commands.py repair mistral controlled_md -a 1 -over

@REM python ./scripts/run_commands.py clean_generations mistral controlled_md -a 2
@REM python ./scripts/run_commands.py test mistral controlled_md -a 2 -over

@REM python ./scripts/run_commands.py repair mistral controlled -a 1 -over

@REM python ./scripts/run_commands.py clean_generations mistral controlled -a 2
@REM python ./scripts/run_commands.py test mistral controlled -a 2 -over

@REM python ./scripts/run_commands.py repair mistral controlled_md -a 2

@REM python ./scripts/run_commands.py clean_generations mistral controlled_md -a 3
@REM python ./scripts/run_commands.py test mistral controlled_md -a 3

@REM python ./scripts/run_commands.py translate mistral via_description -a 1

@REM python ./scripts/run_commands.py translate mistral via_description_1_shot -a 1 -over


python ./scripts/run_commands.py repair mistral controlled_md -a 1 -d bithacks -over
python ./scripts/run_commands.py clean_generations mistral controlled_md -a 2 -d bithacks -over
python ./scripts/run_commands.py test mistral controlled_md -a 2 -d bithacks -over
python ./scripts/run_commands.py repair mistral controlled_md -a 2 -d bithacks -over
python ./scripts/run_commands.py clean_generations mistral controlled_md -a 3 -d bithacks -over
python ./scripts/run_commands.py test mistral controlled_md -a 3 -d bithacks -over