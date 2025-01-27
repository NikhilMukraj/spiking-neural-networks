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
- [ ] basic function calling tests
- [ ] bool vars (should be listed in same `vars:` block)
- [x] read neuron model from text file
- [ ] check syntax tree to see if imports are already imported into scope, if they are do not re-import
- [ ] default ligand gated channel
- [ ] neuronal chemical testing
- [ ] kinetics
- [ ] function definition blocks
- [ ] spike train block
