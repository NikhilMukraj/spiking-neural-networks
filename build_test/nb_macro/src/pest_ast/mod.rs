use pest::pratt_parser::PrattParser;


#[derive(pest_derive::Parser)]
#[grammar_inline = r#"
integer = @{ ASCII_DIGIT+ }
bool = { "true" | "false" }
decimal = @{ ASCII_DIGIT+ ~ "." ~ ASCII_DIGIT* }
number = @{ decimal | integer }
name_characters = @{ 'a'..'z' | 'A'..'z' | "_" }

name = @{ name_characters ~ (name_characters | integer)* }

struct_call = { name ~ "." ~ name ~ ("(" ~ args ~ ")")? } 

args = {
    (expr | struct_call | function | name | number | bool)? ~ (" "* ~ "," ~ " "* ~ (expr | function | name | number | bool))* ~ ","?
}
function = { name ~ ("(" ~ args ~ ")") }

unary_minus = { "-" }
primary = _{ struct_call | function | number | bool | name | "(" ~ expr ~ ")" }

atom = _{ (not_operator ~ primary) | (unary_minus ~ primary) | primary  }

not_operator = { "!" }

bin_op = _{ 
	greater_than_or_equal | less_than_or_equal | equal | not_equal | 
	greater_than | less_than | and_operator | or_operator |
	add | subtract | multiply | divide | power 
}
	equal = { "==" }
	not_equal = { "!=" }
	greater_than = { ">" }
	greater_than_or_equal = { ">=" }
	less_than = { "<" }
	less_than_or_equal = { "<=" }
	and_operator = { "&&" }
	or_operator = { "||" }
	add = { "+" }
	subtract = { "-" }
	multiply = { "*" }
	divide = { "/" }
	power = { "^" }

expr = { atom ~ (bin_op ~ atom)* }

diff_eq_declaration = { "d" ~ (struct_call | name) ~ "/dt" ~ " "* ~ "=" ~ " "* ~ expr }
eq_operator = { "=" | "+=" | "-=" | "*=" | "/=" }
eq_declaration = { (struct_call | name) ~ " "* ~ eq_operator ~ " "* ~ expr }

func_declaration_args = { "(" ~ name ~ (" "* ~ "," ~ " "* ~ name)* ~ ","? ~ ")" }
func_declaration = { (struct_call | name) ~ func_declaration_args ~ " "* ~ "=" ~ " "* ~ expr }

struct_call_execution = { name ~ "." ~ name ~ ("(" ~ args ~ ")") }

WHITESPACE = _{ " " }

signed_number = @{ unary_minus? ~ number }
variables_block = _{ "vars:" ~ " "* ~ name ~ (" "* ~ "," ~ " "* ~ name)* ~ ","? }
variables_assignment = { name ~ " "* ~ "=" ~ " "* ~ (signed_number | bool) }
variables_with_assignment = _{ 
	"vars:" ~ (" " | "\n")* ~ variables_assignment ~ 
    ((" " | "\n")* ~ "," ~ (" " | "\n")* ~ variables_assignment)* ~ 
    ","? 
}

struct_assignment = { name ~ " "* ~ "=" ~ " "* ~ name }
ion_channels_def_with_assignment = _{ 
	"ion_channels:" ~ " "* ~ struct_assignment ~ 
    (" "* ~ "," ~ " "* ~ struct_assignment)* ~ 
    ","? 
}

eq_assignments = _{ eq_declaration ~ (NEWLINE* ~ eq_declaration)* ~ NEWLINE? }

else_body = { "[else]\n" ~ WHITESPACE* ~ assignments }
if_body = { assignments }
else_if_body = { "[elseif]" ~ WHITESPACE* ~ expr ~ WHITESPACE* ~ "[then]\n" ~ WHITESPACE* ~ assignments }
if_statement = { 
	"[if]" ~ WHITESPACE* ~ expr ~ WHITESPACE* ~ "[then]\n" 
	~ WHITESPACE* ~ if_body ~ (else_if_body ~ WHITESPACE*)+? ~ WHITESPACE* ~ else_body? 
	~ NEWLINE? ~ "[end]" 
}

assignment = _{ func_declaration | diff_eq_declaration | eq_declaration | struct_call_execution | if_statement  }
assignments = _{ assignment ~ (NEWLINE* ~ assignment)* ~ NEWLINE? }

type_def = { "type:" ~ " "* ~ name ~ NEWLINE+ }
vars_with_default_def  = { variables_with_assignment ~ NEWLINE+ }
kinetics_def = { "kinetics:" ~ " "* ~ name ~ " "* ~ "," ~ " "* ~ name ~ NEWLINE+ }
on_spike_def = { "on_spike:" ~ " "* ~ (NEWLINE+ | " "+ | "") ~ assignments }
on_iteration_def = { "on_iteration:" ~ " "* ~ (NEWLINE+ | " "+ | "") ~ assignments }
on_electrochemical_iteration_def = { "on_electrochemical_iteration:" ~ " "* ~ (NEWLINE+ | " "+ | "") ~ assignments }
spike_detection_def = { "spike_detection:" ~ " "* ~ expr ~ NEWLINE+ }
ion_channels_def = { ion_channels_def_with_assignment ~ NEWLINE+ }
receptors_param_def = { "receptors:" ~ " "* ~ name ~ NEWLINE+ }

neuron_definition = {
    "[neuron]" ~ NEWLINE ~ 
	(
		on_iteration_def | type_def | kinetics_def | on_spike_def | spike_detection_def | 
		vars_with_default_def |  ion_channels_def | on_electrochemical_iteration_def | 
		receptors_param_def
	){5,} ~ 
	"[end]"
}

refractoriness_definiton = { "neural_refractorienss:" ~ " "* ~ name ~ NEWLINE+ }

spike_train_definition = {
    "[spike_train]" ~ NEWLINE ~ 
	(
		on_iteration_def | type_def | vars_with_default_def | 
		neurotransmitter_def | single_kinetics_def | refractoriness_definiton
	){3,} ~ 
	"[end]"
}

effect_definition = { "effect:" ~ " "* ~ expr ~ NEWLINE+ }

neural_refractoriness_definition = {
	"[neural_refractoriness]" ~ NEWLINE ~
	(
		type_def | vars_with_default_def | effect_definition
	){2,} ~
	"[end]"
}

gating_variables_block = _{ "gating_vars:" ~ " "* ~ name ~ (" "* ~ "," ~ " "* ~ name)* ~ ","? }
gating_variables_def = { gating_variables_block ~ NEWLINE+ }

ion_channel_definition = {
	"[ion_channel]" ~ NEWLINE ~ 
	(gating_variables_def | type_def | vars_with_default_def | on_iteration_def){3,} ~
	"[end]"
}

neurotransmitter_kinetics_definition= {"[neurotransmitter_kinetics]" ~ NEWLINE ~ (type_def | vars_with_default_def | on_iteration_def){3,} ~ "[end]" }
receptor_kinetics_definition = {"[receptor_kinetics]" ~ NEWLINE ~ (type_def | vars_with_default_def | on_iteration_def){3,} ~ "[end]" }

receptor_vars_def = { "receptors:" ~ WHITESPACE* ~ name ~ (" "* ~ "," ~ " "* ~ name)* ~ NEWLINE+ }
receptor_block = {
	neurotransmitter_def ~ NEWLINE* ~ receptor_vars_def{0,1} ~ NEWLINE* ~ vars_with_default_def ~ 
	NEWLINE* ~ on_iteration_def ~ NEWLINE*
}

single_kinetics_def = { "kinetics:" ~ " "* ~ name ~ NEWLINE+ }
neurotransmitter_def = { "neurotransmitter:" ~ " "* ~ name ~ NEWLINE+ }
receptors_definition = {
	"[receptors]" ~ NEWLINE+ ~ 
		(
			type_def | vars_with_default_def | single_kinetics_def |
			receptor_block+
		){2,4} ~ WHITESPACE* ~ "[end]"
}

full = _{
	SOI ~ NEWLINE* ~ ((
		neuron_definition | spike_train_definition | neural_refractoriness_definition | 
		ion_channel_definition | neurotransmitter_kinetics_definition | receptor_kinetics_definition |
        receptors_definition
	)+ ~ NEWLINE*
	)* ~ EOI
}
"#]
pub struct ASTParser;

lazy_static::lazy_static! {
    pub static ref PRATT_PARSER: PrattParser<Rule> = {
        use pest::pratt_parser::{Assoc::*, Op};
        use Rule::*;

        PrattParser::new()
            .op(
                Op::infix(equal, Left) | Op::infix(not_equal, Left) | Op::infix(greater_than, Left) |
                Op::infix(greater_than_or_equal, Left) | Op::infix(less_than, Left) | 
                Op::infix(less_than_or_equal, Left) | Op::infix(and_operator, Left) | 
                Op::infix(or_operator, Left)
            )
            .op(Op::infix(add, Left) | Op::infix(subtract, Left))
            .op(Op::infix(multiply, Left) | Op::infix(divide, Left))
			.op(Op::infix(power, Left))
            .op(Op::prefix(unary_minus) | Op::prefix(not_operator))
    };
}
