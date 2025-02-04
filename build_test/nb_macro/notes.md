# notes

## general

- default ligand gated channel is just neurotransmitter $X$ with ligand gated channel $I_{syn} = g_X r_X (V - E_X)$
- gpu receptors refactor where g, r, E, etc are not shared there's just an indivual r ampa, r nmda, g ampa, g nmda, etc

## todo

- [x] ion channel defaults
  - [x] basic gating variables defaults in ion channels
- [ ] `(struct_call | name)` in front of declarations to edit struct variables
  - [ ] in assignments add a `struct_call_execution` to or where it execution is `_{ struct_call }`
    - eventually change to seperate struct_function_call
  - [ ] calling structs functions
  - [ ] struct var call versus struct func call in declaration
  - [ ] rename assignments to calls (call assignments for current assignments)
- [ ] if statements (handled by statement blocks)
  - `if_statement = { "[if]" ~ WHITESPACE* ~ expr ~ WHITESPACE* ~ "[then]\n" ~ WHITESPACE* ~ assignments ~ NEWLINE? ~ "[end]" }`
  - optional else if and else in between if and end
  - option within if statement ast
    - could have vec of conditionals and vec of vec of declarations and iterate through each one
- [ ] basic function calling tests
- [ ] **bool vars (should be listed in same `vars:` block)**
- [ ] `+=`, `-=`, `*=`, `/=`
- [x] read neuron model from text file
- [ ] check syntax tree to see if imports are already imported into scope, if they are do not re-import
- [ ] remove vars without defaults, all vars should have associated defaults
- [ ] `to_code` abstraction in generation
- [ ] rename ligand gates to receptors
- [ ] receptors
  - [ ] ionotropic receptors
    - [ ] default ligand gated channel
      - default operation with receptors is just to add receptor currents
      - should have option to customize receptors
  - [ ] metabotropic receptors
- [ ] neuronal chemical testing
- [ ] kinetics
  - [x] neurotransmitter kinetics
    - declare neurotransmitter kinetics name
    - declare vars
    - assignment blocks
    - `neurotransmitter_kinetics = {"[neurotransmitter_kinetics]" ~ NEWLINE ~ (vars_with_default_def | on_iteration_def){2,} ~ "[end]" }`
    - make sure to add to full def and make sure that both vars and on iteration are defined before parsing
  - [x] receptor kinetics
    - declare receptor kinetics name
    - declare vars
    - assignment blocks
    - `receptor_kinetics = {"[receptor_kinetics]" ~ NEWLINE ~ (vars_with_default_def | on_iteration_def){2,} ~ "[end]" }`
  - [ ] default impl method for default kinetics
  - [ ] default impl trait maybe
- [ ] **in `fn generate_x(pairs: Pairs<Rule>) -> Result<Def>`, create custom errors for when definition is not present**
- [ ] function definition blocks
- [ ] spike train block
- [ ] eventually remove vars declaration without vars
- [ ] comments/docs in/around blocks
- [ ] gpu implementations
