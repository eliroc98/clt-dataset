# Dataset construction
I want to build a diverse dataset composed by many minimally-changing prompts.

At the moment, I have a large collection of prompts characterised by the following components:

- task specification: this include both tasks and constraints
- options: what is practically asked

For example, in the prompt "Write a long email template that invites a group of participants to a meeting, with at least 500 words. The email must include the keywords \"correlated\" and \"experiencing\" and should not use any commas." there are 4 parts:

- task specification template (1): Write [option = a long email template that invites a group of participants to a meeting]
- task specification template (2): with at least [option = 500] words
- task specification template (3): The [option that is related to the previous = email] must include the keywords "[option = correlated]" and "[option = experiencing]"
- task specification template (4): should not use any [option = commas]

Here are the steps to implement a (semi)syntetic dataset generation:

1. for each prompt in each dataset, extract task specification templates that are in the prompt and the options
2. store the templates in a tidy dataset, where the task type is specified (according to the taxonomy)
3. store the options in a dataset, so that one can put them back into the templates. Consider that I want sensical prompts, so I need to store which type of tasks and templates can be completed with the options
4. (optional) augment the options. For example, in the number of words task ("... with at least 500 words"), the option = 500 can be expanded to any integer number (which however must make sense with the request: if I am asking for a paragraph, I should not constrain the generation to 2 words.)
4. synthetically generate the dataset. I need to have control over the length of the final prompt, thus storing the length of each part (task templates and options) can be handy. Morevoer, I would like to specify how many task templates to use together (1 = 1 task template. Obviously, some tasks like number of words do not make sense alone; 2 = text generation + number of words)
5. analyse the density of the dataset. The aim here is to produce as many combinations as possible of every task template and options (eg if I have 2 task templates and 4 options that make sense with those templates, then I would hypothetically like to have 2*4 synthetic prompts). However, since not all options make sense with all templates, I know I won't be able to reach maximum optimal density: I will probably have feasible density vs full density, with feasible density being a subset of possible density. I would like also to have an idea of how much constraining for prompt length is impacting the coverage of all possible sensical combinations.