use std::collections::HashMap;
use std::fs::File;
use std::io::{Error, ErrorKind, Write};
use std::{io::Result, env, fs};
use pest::iterators::{Pair, Pairs};
use pest::pratt_parser::PrattParser;
use pest::Parser;


#[derive(pest_derive::Parser)]
#[grammar = "ast.pest"]
pub struct ASTParser;

lazy_static::lazy_static! {
    // static ref PRATT_PARSER: PrattParser<Rule> = {
    //     use pest::pratt_parser::{Assoc::*, Op};
    //     use Rule::*;

    //     PrattParser::new()
    //         .op(Op::infix(add, Left) | Op::infix(subtract, Left))
    //         .op(Op::infix(multiply, Left) | Op::infix(divide, Left) | Op::infix(power, Left))
    //         .op(Op::prefix(unary_minus))
    // };

    static ref PRATT_PARSER: PrattParser<Rule> = {
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
            .op(Op::infix(multiply, Left) | Op::infix(divide, Left) | Op::infix(power, Left))
            .op(Op::prefix(unary_minus) | Op::prefix(not_operator))
    };
}

#[derive(Debug)]
pub enum AST {
    Number(f32),
    Name(String),
    UnaryMinus(Box<AST>),
    NotOperator(Box<AST>),
    BinOp {
        lhs: Box<AST>,
        op: Op,
        rhs: Box<AST>,
    },
    Function {
        name: String,
        args: Vec<Box<AST>>
    },
    EqAssignment {
        name: String,
        expr: Box<AST>,
    },
    DiffEqAssignment {
        name: String,
        expr: Box<AST>,
    },
    FunctionAssignment {
        name: String,
        args: Vec<String>,
        expr: Box<AST>,
    },
    TypeDefinition(String),
    OnSpike(Vec<Box<AST>>),
    OnIteration(Vec<Box<AST>>),
    SpikeDetection(Box<AST>),
    VariableAssignment {
        name: String,
        value: Option<f32>,
    },
    VariablesAssignments(Vec<Box<AST>>),
}


#[derive(Debug)]
pub enum Op {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    And,
    Or,
}

impl AST {
    pub fn to_string(&self) -> String {
        match self {
            AST::Number(n) => n.to_string(),
            AST::Name(name) => {
                if name == "v" {
                    String::from("self.current_voltage")
                } else if name == "i" {
                    String::from("input_current")
                } else {
                    format!("self.{}", name)
                }
            },
            AST::UnaryMinus(expr) => format!("-{}", expr.to_string()),
            AST::NotOperator(expr) => format!("!{}", expr.to_string()),
            AST::BinOp { lhs, op, rhs } => {
                match op {
                    Op::Add => format!("({} + {})", lhs.to_string(), rhs.to_string()),
                    Op::Subtract => format!("({} - {})", lhs.to_string(), rhs.to_string()),
                    Op::Multiply => format!("({} * {})", lhs.to_string(), rhs.to_string()),
                    Op::Divide => format!("({} / {})", lhs.to_string(), rhs.to_string()),
                    Op::Power => format!("({}.powf({}))", lhs.to_string(), rhs.to_string()),
                    Op::Equal => format!("{} == {}", lhs.to_string(), rhs.to_string()),
                    Op::NotEqual => format!("{} != {}", lhs.to_string(), rhs.to_string()),
                    Op::GreaterThan => format!("{} > {}", lhs.to_string(), rhs.to_string()),
                    Op::GreaterThanOrEqual => format!("{} >= {}", lhs.to_string(), rhs.to_string()),
                    Op::LessThan => format!("{} < {}", lhs.to_string(), rhs.to_string()),
                    Op::LessThanOrEqual => format!("{} <= {}", lhs.to_string(), rhs.to_string()),
                    Op::And => format!("{} && {}", lhs.to_string(), rhs.to_string()),
                    Op::Or => format!("{} || {}", lhs.to_string(), rhs.to_string()),
                }
            }
            AST::Function { name, args } => {
                format!(
                    "{}({})",
                    name, 
                    args.iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                    )
            },
            AST::EqAssignment { name, expr } => {
                let name = if name == "v" {
                    String::from("self.current_voltage")
                } else {
                    format!("self.{}", name)
                };

                format!("{} = {};", name, expr.to_string())
            },
            AST::DiffEqAssignment { name, expr } => {
                format!("let d{} = ({}) * self.dt;", name, expr.to_string())
            },
            AST::FunctionAssignment{ name, args, expr } =>{
                format!(
                    "{}({}) = {}",
                    name, 
                    args.iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<String>>()
                        .join(", "),
                    expr.to_string(),
                )
            },
            AST::TypeDefinition(string) => string.clone(),
            AST::OnSpike(assignments) => {
                assignments.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
                    .join("\n")
            },
            AST::OnIteration(assignments) => {
                assignments.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
                    .join("\n\t\t")
            },
            AST::SpikeDetection(expr) => { expr.to_string() },
            AST::VariableAssignment { name, value } => {
                let value = match value {
                    Some(x) => x.to_string(),
                    None => String::from("None"),
                };

                format!("{} = {}", name, value)
            },
            AST::VariablesAssignments(assignments) => {
                let assignments_string = assignments.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
                    .join("\n\t");

                format!("vars:\n\t{}", assignments_string)
            }
        }
    }
}

fn add_indents(input: &str, indent: &str) -> String {
    input.lines()
        .map(|line| format!("{}{}", indent, line))
        .collect::<Vec<String>>()
        .join("\n")
}

pub struct NeuronDefinition {
    type_name: AST,
    vars: AST,
    on_spike: AST,
    on_iteration: AST,
    spike_detection: AST,
}

impl NeuronDefinition {
    // eventually adapt for documentation to be integrated
    // for now use default ligand gates and neurotransmitter implementation
    // if defaults come with vars assignment then add default trait
    // if neurotransmitter kinetics and receptor kinetics specified then
    // create default_impl() function
    fn to_code(&self) -> String {
        let iterate_and_spike_base = "use spiking_neural_networks::neuron::iterate_and_spike_traits::IterateAndSpikeBase;";
        let necessary_imports = vec![
            "CurrentVoltage", "GapConductance", "GaussianFactor", "LastFiringTime", "IsSpiking",
            "IterateAndSpike", "GaussianParameters", "LigandGatedChannels", 
            "Neurotransmitters", "NeurotransmitterKinetics", "ReceptorKinetics",
            "NeurotransmitterConcentrations",
        ];
        let necessary_imports = format!(
            "use spiking_neural_networks::neuron::iterate_and_spike::{{{}}};",
            necessary_imports.join(", ")
        );

        let neurotransmitter_kinetics = "ApproximateNeurotransmitter";
        let receptor_kinetics = "ApproximateReceptor";

        let kinetics_import = format!(
            "use spiking_neural_networks::neuron::iterate_and_spike::{{{}, {}}};",
            neurotransmitter_kinetics,
            receptor_kinetics,
        );

        let import_statement = format!(
            "{}\n{}\n{}\n\n",
            iterate_and_spike_base,
            necessary_imports,
            kinetics_import,
        );

        let macros = "#[derive(Debug, Clone, IterateAndSpikeBase)]";
        let header = format!(
            "pub struct {}<T: NeurotransmitterKinetics, R: ReceptorKinetics> {{", 
            self.type_name.to_string(),
        );
        let mut fields = match &self.vars {
            AST::VariablesAssignments(variables) => {
                variables
                    .iter()
                    .map(|i| {
                        let var_name = match i.as_ref() {
                            AST::VariableAssignment { name, .. } => name,
                            _ => unreachable!(),
                        };

                        format!("{}: f32", var_name)
                    })
                    .collect::<Vec<String>>()
            },
            _ => unreachable!()
        };
        let current_voltage_field = String::from("current_voltage: f32");
        let dt_field = String::from("dt: f32");
        let c_m_field = String::from("c_m: f32");
        let gap_conductance_field = String::from("gap_conductance: f32");
        let is_spiking_field = String::from("is_spiking: bool");
        let last_firing_time_field = String::from("last_firing_time: Option<usize>");
        let gaussian_field = String::from("gaussian_params: GaussianParameters");
        let neurotransmitter_field = String::from("synaptic_neurotransmitters: Neurotransmitters<T>");
        let ligand_gates_field = String::from("ligand_gates: LigandGatedChannels<R>");

        fields.insert(0, current_voltage_field);
        fields.push(gap_conductance_field);
        fields.push(dt_field);
        fields.push(c_m_field);
        fields.push(is_spiking_field);
        fields.push(last_firing_time_field);
        fields.push(gaussian_field);
        fields.push(neurotransmitter_field);
        fields.push(ligand_gates_field);

        let fields = format!("\t{},", fields.join(",\n\t"));

        let handle_spiking_header = "fn handle_spiking(&mut self) -> bool {";
        let handle_is_spiking_calc = format!("\tself.is_spiking = {};", self.spike_detection.to_string());
        let handle_spiking_check = "\tif self.is_spiking {";
        let handle_spiking_function = format!("\t\t{}", self.on_spike.to_string());

        let handle_spiking = format!(
            "{}\n{}\n{}\n{}\n\t}}\n\n\tself.is_spiking\n}}", 
            handle_spiking_header, 
            handle_is_spiking_calc,
            handle_spiking_check, 
            handle_spiking_function
        );

        let on_iteration_assignments = self.on_iteration.to_string();

        let changes = match &self.on_iteration {
            AST::OnIteration(assignments) => {
                let mut assignments_strings = vec![];

                for i in assignments {
                    match i.as_ref() {
                        AST::DiffEqAssignment { name, .. } => {
                            let change_string = if name == "v" {
                                format!("self.current_voltage += dv;")
                            } else {
                                format!("self.{} += d{}", name, name)
                            };

                            assignments_strings.push(change_string);
                        }
                        _ => {}
                    }
                }

                assignments_strings.join("\t\n")
            },
            _ => unreachable!()
        };

        let get_concentrations_header = "fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations {";
        let get_concentrations_body = "self.synaptic_neurotransmitters.get_concentrations()";
        let get_concentrations_function = format!("{}\n\t{}\n}}", get_concentrations_header, get_concentrations_body);

        let handle_neurotransmitter_conc = "self.synaptic_neurotransmitters.apply_t_changes(self.current_voltage);";
        let handle_spiking_call = "self.handle_spiking()";
        let iteration_header = "fn iterate_and_spike(&mut self, input_current: f32) -> bool {";
        let iteration_body = format!(
            "\n\t{}\n\t{}\n\t{}\n\t{}", 
            on_iteration_assignments, 
            changes, 
            handle_neurotransmitter_conc,
            handle_spiking_call,
        );
        let iteration_function = format!("{}{}\n}}", iteration_header, iteration_body);

        let iteration_with_neurotransmission_start = "fn iterate_with_neurotransmitter_and_spike(";
        let iteration_with_neurotransmission_args = vec![
            "&mut self", 
            "input_current: f32",
            "t_total: &NeurotransmitterConcentrations",
        ];
        let iteration_with_neurotransmitter_header = format!(
            "{}\n\t{},\n) -> bool {{", 
            iteration_with_neurotransmission_start, 
            iteration_with_neurotransmission_args.join(",\n\t"),
        );

        let ligand_gates_update = "self.ligand_gates.update_receptor_kinetics(t_total);";
        let ligand_gates_set_current = "self.ligand_gates.set_receptor_currents(self.current_voltage);";

        let update_with_receptor_current = "self.current_voltage += self.ligand_gates.get_receptor_currents(self.dt, self.c_m);";

        let iteration_with_neurotransmission_body = format!(
            "\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}",
            ligand_gates_update,
            ligand_gates_set_current,
            on_iteration_assignments,
            changes,
            update_with_receptor_current,
            handle_neurotransmitter_conc,
            handle_spiking_call,
        );

        let iteration_with_neurotransmission_function = format!(
            "{}\n{}\n}}", 
            iteration_with_neurotransmitter_header,
            iteration_with_neurotransmission_body,
        );

        let impl_header = format!(
            "impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> {}<T, R> {{", 
            self.type_name.to_string()
        );
        let impl_body = add_indents(&handle_spiking, "\t");
        let impl_functions = format!("{}\n{}\n}}", impl_header, impl_body);

        let impl_header_iterate_and_spike = format!(
            "impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for {}<T, R> {{", 
            self.type_name.to_string()
        );
        let impl_iterate_and_spike_body = format!(
            "{}\n\n{}\n\n{}\n",
            get_concentrations_function,
            iteration_function,
            iteration_with_neurotransmission_function,
        );
        let impl_iterate_and_spike_body = add_indents(&impl_iterate_and_spike_body, "\t");
        let impl_iterate_and_spike = format!(
            "{}\n{}\n}}", 
            impl_header_iterate_and_spike, 
            impl_iterate_and_spike_body,
        );

        format!(
            "{}\n{}\n{}\n{}\n}}\n\n{}\n\n{}\n", 
            import_statement,
            macros, 
            header, 
            fields, 
            impl_functions, 
            impl_iterate_and_spike,
        )
    }
}

// pub struct IonChannelDefinition {
//     type_name: AST,
//     vars: AST,
//     on_iteration: AST,
// }

// then try writing rust code from ast
pub fn parse_expr(pairs: Pairs<Rule>) -> AST {
    PRATT_PARSER
        .map_primary(|primary| match primary.as_rule() {
            Rule::number => AST::Number(primary.as_str().parse::<f32>().unwrap()),
            Rule::name => AST::Name(String::from(primary.as_str())),
            Rule::expr => parse_expr(primary.into_inner()),
            Rule::function => {
                let mut inner_rules = primary.into_inner();

                let name: String = String::from(inner_rules.next()
                    .expect("Could not get function name").as_str()
                );

                let args: Vec<Box<AST>> = inner_rules.next()
                    .expect("No arguments found")
                    .into_inner()
                    .map(|i| Box::new(parse_expr(i.into_inner())))
                    .collect();
                
                AST::Function { name: name, args: args }
            },
            rule => unreachable!("AST::parse expected atom, found {:?}", rule),
        })
        .map_infix(|lhs, op, rhs| {
            let op = match op.as_rule() {
                Rule::add => Op::Add,
                Rule::subtract => Op::Subtract,
                Rule::multiply => Op::Multiply,
                Rule::divide => Op::Divide,
                Rule::power => Op::Power,
                rule => unreachable!("AST::parse expected (non boolean) infix operation, found {:?}", rule),
            };
            AST::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
            }
        })
        .map_prefix(|op, rhs| match op.as_rule() {
            Rule::unary_minus => AST::UnaryMinus(Box::new(rhs)),
            _ => unreachable!(),
        })
        .parse(pairs)
}

pub fn bool_parse_expr(pairs: Pairs<Rule>) -> AST {
    PRATT_PARSER
        .map_primary(|primary| match primary.as_rule() {
            Rule::number => AST::Number(primary.as_str().parse::<f32>().unwrap()),
            Rule::name => AST::Name(String::from(primary.as_str())),
            Rule::expr => bool_parse_expr(primary.into_inner()),
            Rule::function => {
                let mut inner_rules = primary.into_inner();

                let name: String = String::from(inner_rules.next()
                    .expect("Could not get function name").as_str()
                );

                let args: Vec<Box<AST>> = inner_rules.next()
                    .expect("No arguments found")
                    .into_inner()
                    .map(|i| Box::new(bool_parse_expr(i.into_inner())))
                    .collect();
                
                AST::Function { name: name, args: args }
            },
            rule => unreachable!("AST::parse expected atom, found {:?}", rule),
        })
        .map_infix(|lhs, op, rhs| {
            let op = match op.as_rule() {
                Rule::add => Op::Add,
                Rule::subtract => Op::Subtract,
                Rule::multiply => Op::Multiply,
                Rule::divide => Op::Divide,
                Rule::power => Op::Power,
                Rule::equal => Op::Equal,
                Rule::not_equal => Op::NotEqual,
                Rule::greater_than => Op::GreaterThan,
                Rule::greater_than_or_equal => Op::GreaterThanOrEqual,
                Rule::less_than => Op::LessThan,
                Rule::less_than_or_equal => Op::LessThanOrEqual,
                Rule::and_operator => Op::And,
                Rule::or_operator => Op::Or,
                rule => unreachable!("AST::parse expected infix operation, found {:?}", rule),
            };
            AST::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
            }
        })
        .map_prefix(|op, rhs| match op.as_rule() {
            Rule::unary_minus => AST::UnaryMinus(Box::new(rhs)),
            Rule::not_operator => AST::NotOperator(Box::new(rhs)),
            _ => unreachable!(),
        })
        .parse(pairs)
}

pub fn parse_declaration(pair: Pair<Rule>) -> AST {
    match pair.as_rule() {
        Rule::diff_eq_declaration => {
            let mut inner_rules = pair.into_inner();

            let name: String = String::from(inner_rules.next()
                .expect("Could not get function name").as_str()
            );

            let expr: Box<AST> = Box::new(
                parse_expr(
                    inner_rules.next()
                        .expect("No arguments found")
                        .into_inner()
                )
            );

            AST::DiffEqAssignment { name: name, expr: expr }
        },
        Rule::eq_declaration => {
            let mut inner_rules = pair.into_inner();

            let name: String = String::from(inner_rules.next()
                .expect("Could not get function name").as_str()
            );

            let expr: Box<AST> = Box::new(
                parse_expr(
                    inner_rules.next()
                        .expect("No arguments found")
                        .into_inner()
                )
            );

            AST::EqAssignment { name: name, expr: expr }
        },
        Rule::func_declaration => {
            let mut inner_rules = pair.into_inner();
            let name = String::from(inner_rules.next().unwrap().as_str());

            let args = inner_rules.next().unwrap()
                .into_inner()
                .map(|arg| String::from(arg.as_str()))
                .collect::<Vec<String>>();

            let expr = Box::new(parse_expr(inner_rules.next().unwrap().into_inner()));

            AST::FunctionAssignment {
                name,
                args,
                expr,
            }
        }
        rule => unreachable!("Unexpected declaration, found {:#?}", rule),
    }
}

fn main() -> Result<()> {
    let mut filename = String::from("");

    for (key, value) in env::vars() {
        if key == "filename" {
            filename = value;
        }
    }

    if filename == "" {
        return Ok(())
    }

    let contents = fs::read_to_string(filename)?;

    // handle variables
    // handle continous detection
    // try code generation (assume default ligands)
    // default functions
    // runge kutta

    // handle ion channels (handle builtin ion channels) 
    // (could import with name prefixed as DefaultChannel or something)
    // handle ligand gates
    // neurotransmitter and approximate kinetics
    // handling function if statements and boolean vars

    match ASTParser::parse(Rule::neuron_definition, &contents) {
        Ok(pairs) => {
            let mut definitions: HashMap<String, AST> = HashMap::new();

            for pair in pairs {
                let (key, current_ast) = match pair.as_rule() {
                    Rule::type_def => {
                        (
                            String::from("type"), 
                            AST::TypeDefinition(
                                String::from(pair.into_inner().next().unwrap().as_str())
                            )
                        )
                    },
                    Rule::on_iteration_def => {
                        let inner_rules = pair.into_inner();

                        (
                            String::from("on_iteration"),
                            AST::OnIteration(
                                inner_rules
                                .map(|i| Box::new(parse_declaration(i)))
                                .collect::<Vec<Box<AST>>>()
                            )
                        )
                    },
                    Rule::on_spike_def => {
                        let inner_rules = pair.into_inner();

                        (
                            String::from("on_spike"),
                            AST::OnSpike(
                                inner_rules
                                .map(|i| Box::new(parse_declaration(i)))
                                .collect::<Vec<Box<AST>>>()
                            )
                        )
                    },
                    Rule::spike_detection_def => {
                        (
                            String::from("spike_detection"),
                            AST::SpikeDetection(Box::new(bool_parse_expr(pair.into_inner())))
                        )
                    }
                    Rule::vars_def => {
                        // if no defaults then just assume assingment is None
                        // in order to prevent duplicate, key should be "vars"
                        let inner_rules = pair.into_inner();

                        let assignments: Vec<Box<AST>> = inner_rules
                            .map(|i| Box::new(AST::VariableAssignment { 
                                name: String::from(i.as_str()), 
                                value: None,
                            }))
                            .collect();

                        println!("{:#?}", assignments);

                        (
                            String::from("vars"),
                            AST::VariablesAssignments(assignments)
                        )
                    },
                    Rule::vars_with_default_def => {
                        // assignment should be just a number
                        // in order to prevent duplicate, key should be "vars"

                        let inner_rules = pair.into_inner();

                        let assignments: Vec<Box<AST>> = inner_rules 
                            .map(|i| {
                                let mut nested_rule = i.into_inner();

                                Box::new(AST::VariableAssignment { 
                                    name: String::from(nested_rule.next().unwrap().as_str()), 
                                    value: Some(
                                        nested_rule.next()
                                            .unwrap()
                                            .as_str()
                                            .parse::<f32>()
                                            .unwrap()
                                        ), 
                                })
                            })
                            .collect(); 

                        (
                            String::from("vars"),
                            AST::VariablesAssignments(assignments)
                        )
                    },
                    definition => unreachable!("Unexpected definiton: {:#?}", definition)
                };

                if definitions.contains_key(&key) {
                    return Err(
                        Error::new(
                            ErrorKind::InvalidInput, format!("Duplicate definition found: {}", key),
                        )
                    )
                }

                definitions.insert(key, current_ast);
            }

            // neuron definition as part of ast enum?

            let neuron = NeuronDefinition {
                type_name: definitions.remove("type").unwrap(),
                vars: definitions.remove("vars").unwrap(),
                spike_detection: definitions.remove("spike_detection").unwrap(),
                on_iteration: definitions.remove("on_iteration").unwrap(),
                on_spike: definitions.remove("on_spike").unwrap()
            };

            let mut file = File::create("neuron_file.rs")?;
            file.write_all(neuron.to_code().as_bytes())?;
        }
        Err(e) => {
            eprintln!("Parse failed: {:?}", e);
        }
    }

    Ok(())
}
