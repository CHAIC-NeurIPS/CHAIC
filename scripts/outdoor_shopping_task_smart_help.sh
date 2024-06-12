port=11000
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name outdoor_shopping_task_random_helper \
--run_id smart_help_train_collection_2 \
--port $port \
--agents plan_agent plan_agent \
--plan_mode default random \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--cot \
--lm_id gpt-4 \
--max_frames 3000 \
--data_prefix dataset/train_dataset/outdoor_shopping \
--screen_size 512 \
--debug \
--gt_mask \
--smart_help \
--gt_behavior

ps ux | grep port\ $port | awk {'print $2'} | xargs kill