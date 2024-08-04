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
    StructCall {
        name: String,
        attribute: String,
        args: Option<Vec<Box<AST>>>,
    },
    StructAssignment {
        name: String,
        type_name: String,
    },
    StructAssignments(Vec<Box<AST>>),
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
    GatingVariables(Vec<String>),
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
            AST::StructCall { name, attribute, args } => {
                format!(
                    "self.{}.{}{}", 
                    name, 
                    attribute,
                    match args {
                        Some(args) => {
                            args.iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<String>>()
                                .join(", ")
                        },
                        None => String::from(""),
                    }
                )
            }
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
            AST::GatingVariables(vars) => {
                format!("gating_vars: {}", vars.join(", "))
            },
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
            },
            AST::StructAssignment { name, type_name } => {
                format!("{} = {}", name, type_name)
            },
            AST::StructAssignments(assignments) => {
                let assignments_string = assignments.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
                    .join("\n\t");

                format!("structs:\n\t{}", assignments_string)
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
    ion_channels: Option<AST>,
}

impl NeuronDefinition {
    // eventually adapt for documentation to be integrated
    // for now use default ligand gates and neurotransmitter implementation
    // if defaults come with vars assignment then add default trait
    // if neurotransmitter kinetics and receptor kinetics specified then
    // create default_impl() function
    fn to_code(&self) -> (Vec<String>, String) {
        let neurotransmitter_kinetics = "ApproximateNeurotransmitter";
        let receptor_kinetics = "ApproximateReceptor";

        let kinetics_import = format!(
            "use spiking_neural_networks::neuron::iterate_and_spike::{{{}, {}}};",
            neurotransmitter_kinetics,
            receptor_kinetics,
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

        let ion_channels = match &self.ion_channels {
            Some(AST::StructAssignments(variables)) => {
                variables.iter()
                    .map(|i| {
                        let (var_name, type_name) = match i.as_ref() {
                            AST::StructAssignment { name, type_name } => (name, type_name),
                            _ => unreachable!(),
                        };

                        format!("{}: {}", var_name, type_name)
                    })
                    .collect::<Vec<String>>()
            },
            None => vec![],
            _ => unreachable!()
        };

        ion_channels.iter()
            .for_each(|i| fields.push(i.clone()));

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

        (
            vec![String::from(kinetics_import)],
            format!(
                "{}\n{}\n{}\n}}\n\n{}\n\n{}\n", 
                macros, 
                header, 
                fields, 
                impl_functions, 
                impl_iterate_and_spike,
            )
        )
    }
}

pub fn generate_neuron(pairs: Pairs<Rule>) -> Result<NeuronDefinition> {
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
                    AST::SpikeDetection(Box::new(parse_bool_expr(pair.into_inner())))
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
            Rule::ion_channels_def => {
                let inner_rules = pair.into_inner();

                let assignments: Vec<Box<AST>> = inner_rules 
                    .map(|i| {
                        let mut nested_rule = i.into_inner();

                        Box::new(AST::StructAssignment { 
                            name: String::from(nested_rule.next().unwrap().as_str()), 
                            type_name: String::from(
                                nested_rule.next()
                                    .unwrap()
                                    .as_str()
                            )
                        })
                    })
                    .collect(); 

                (
                    String::from("ion_channels"),
                    AST::StructAssignments(assignments)
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
        on_spike: definitions.remove("on_spike").unwrap(),
        ion_channels: definitions.remove("ion_channels"),
    };

    Ok(neuron)
}

pub struct IonChannelDefinition {
    type_name: AST,
    vars: AST,
    gating_vars: Option<AST>,
    on_iteration: AST,
}

impl IonChannelDefinition {
    fn get_use_timestep(&self) -> bool {
        match &self.on_iteration {
            AST::OnIteration(assignments) => {
                let mut use_timestep = false;

                for i in assignments {
                    match i.as_ref() {
                        AST::DiffEqAssignment { .. } => { use_timestep = true },
                        _ => {},
                    }
                }

                use_timestep
            },
            _ => unreachable!()
        }
    }

    // for now assume all gating variables default to 0 for a and b
    fn to_code(&self) -> (Vec<String>, String) {
        let mut imports = vec![];

        let header = format!(
            "#[derive(Debug, Clone, Copy)]\npub struct {} {{", 
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

        let gating_variables = match &self.gating_vars {
            Some(AST::GatingVariables(variables)) => {
                imports.push(
                    String::from(
                        "use spiking_neural_networks::neuron::ion_channels::BasicGatingVariable;"
                    )
                );

                variables.clone()
                    .iter()
                    .map(|i| format!("{}: BasicGatingVariable", i))
                    .collect()
            },
            None => vec![],
            _ => unreachable!()
        };

        for i in gating_variables {
            fields.push(i)
        }

        let current_field = String::from("current: f32");
        fields.push(current_field);

        let fields = format!("\t{},", fields.join(",\n\t"));

        let use_timestep = self.get_use_timestep();

        let get_current = "fn get_current(&self) -> f32 { self.current }";

        let update_current = if use_timestep {
            let update_current_header = "fn update_current(&mut self, voltage: f32, dt: f32) {";
            let on_iteration = &self.on_iteration.to_string();

            let mut lines: Vec<&str> = on_iteration.split('\n').collect();
            let current_line_index = lines.iter().position(|&line| line.starts_with("self.current"));

            let current_assignment = match current_line_index {
                Some(index) => lines.remove(index),
                None => "",
            };

            let update_current_body = add_indents(&lines.join("\n"), "\t");

            let changes = match &self.on_iteration {
                AST::OnIteration(assignments) => {
                    let mut assignments_strings = vec![];
    
                    for i in assignments {
                        match i.as_ref() {
                            AST::DiffEqAssignment { name, .. } => {
                                assignments_strings.push(format!("self.{} += d{}", name, name));
                            }
                            _ => {}
                        }
                    }
    
                    assignments_strings.join("\t\n")
                },
                _ => unreachable!()
            };

            let changes = add_indents(&changes, "\t");

            format!(
                "{}\n{}\n{}\n{}\n}}", 
                update_current_header, 
                update_current_body, 
                changes, 
                current_assignment
            )
        } else {
            let update_current_header = "fn update_current(&mut self, voltage: f32) {";
            let update_current_body = add_indents(&self.on_iteration.to_string(), "\t");
            format!("{}\n{}\n}}", update_current_header, update_current_body)
        };
        
        // if use timestep then header is ionchannel
        // otherwise header is timestepindenpendentionchannel
        let impl_header = if use_timestep {
            format!("impl IonChannel for {} {{", self.type_name.to_string())
        } else {
            format!("impl TimestepIndependentIonChannel for {} {{", self.type_name.to_string())
        };

        if use_timestep {
            imports.push(
                String::from(
                    "use spiking_neural_networks::neuron::ion_channels::IonChannel;"
                )
            );
        } else {
            imports.push(
                String::from(
                    "use spiking_neural_networks::neuron::ion_channels::TimestepIndependentIonChannel;"
                )
            );
        };

        // code may need to be updated if current is assigned using 

        let update_current = add_indents(&update_current, "\t");
        let get_current = add_indents(&get_current, "\t");

        (
            imports, 
            format!(
                "{}\n{}\n}}\n\n{}\n{}\n\n{}\n}}\n", 
                header, 
                fields, 
                impl_header, 
                update_current, 
                get_current
            )
        )
    }
}

pub fn generate_ion_channel(pairs: Pairs<Rule>) -> Result<IonChannelDefinition> {
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
            Rule::gating_variables_def => {
                let inner_rules = pair.into_inner();

                let assignments: Vec<String> = inner_rules 
                    .map(|i| {
                        String::from(i.as_str())
                    })
                    .collect(); 

                (
                    String::from("gating_vars"),
                    AST::GatingVariables(assignments)
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

    let ion_channel = IonChannelDefinition {
        type_name: definitions.remove("type").unwrap(),
        vars: definitions.remove("vars").unwrap(),
        gating_vars: definitions.remove("gating_vars"),
        on_iteration: definitions.remove("on_iteration").unwrap(),
    };

    Ok(ion_channel)
}

// then try writing rust code from ast
pub fn parse_expr(pairs: Pairs<Rule>) -> AST {
    PRATT_PARSER
        .map_primary(|primary| match primary.as_rule() {
            Rule::number => AST::Number(primary.as_str().parse::<f32>().unwrap()),
            Rule::name => AST::Name(String::from(primary.as_str())),
            Rule::expr => parse_expr(primary.into_inner()),
            Rule::struct_call => {
                let mut inner_rules = primary.into_inner();

                let name: String = String::from(
                    inner_rules.next()
                        .expect("Could not get struct name").as_str()
                );

                let attribute: String = String::from(
                    inner_rules.next()
                        .expect("Could not get attribute").as_str()
                );

                let args: Option<Vec<Box<AST>>> = match inner_rules.next() {
                    Some(value) => {
                        Some(
                            value.into_inner()
                            .map(|i| Box::new(parse_expr(i.into_inner())))
                            .collect()
                        )
                    },
                    None => None,
                };
                
                AST::StructCall { name: name, attribute: attribute, args: args }
            }
            Rule::function => {
                let mut inner_rules = primary.into_inner();

                let name: String = String::from(
                    inner_rules.next()
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

pub fn parse_bool_expr(pairs: Pairs<Rule>) -> AST {
    PRATT_PARSER
        .map_primary(|primary| match primary.as_rule() {
            Rule::number => AST::Number(primary.as_str().parse::<f32>().unwrap()),
            Rule::name => AST::Name(String::from(primary.as_str())),
            Rule::expr => parse_bool_expr(primary.into_inner()),
            Rule::struct_call => {
                let mut inner_rules = primary.into_inner();

                let name: String = String::from(
                    inner_rules.next()
                        .expect("Could not get struct name").as_str()
                );

                let attribute: String = String::from(
                    inner_rules.next()
                        .expect("Could not get attribute").as_str()
                );

                let args: Option<Vec<Box<AST>>> = match inner_rules.next() {
                    Some(value) => {
                        Some(
                            value.into_inner()
                            .map(|i| Box::new(parse_bool_expr(i.into_inner())))
                            .collect()
                        )
                    },
                    None => None,
                };
                
                AST::StructCall { name: name, attribute: attribute, args: args }
            },
            Rule::function => {
                let mut inner_rules = primary.into_inner();

                let name: String = String::from(inner_rules.next()
                    .expect("Could not get function name").as_str()
                );

                let args: Vec<Box<AST>> = inner_rules.next()
                    .expect("No arguments found")
                    .into_inner()
                    .map(|i| Box::new(parse_bool_expr(i.into_inner())))
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

// fn insert_at_substring(original: &str, to_find: &str, to_insert: &str) -> Option<String> {
//     if let Some(start) = original.find(to_find) {
//         let mut result = String::new();
//         result.push_str(&original[..start]); // Add the part before the substring
//         result.push_str(to_insert); // Add the string to insert
//         result.push_str(&original[start..]); // Add the part from the substring to the end
//         Some(result)
//     } else {
//         None // Substring not found
//     }
// }

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

    let contents = fs::read_to_string(&filename)?;

    // handle variables
    // handle continous detection
    // try code generation (assume default ligands)

    // handle ion channels (handle builtin ion channels) 
    // handle gating variables
    // (could import with name prefixed as DefaultChannel or something)

    // update ion channel is called before other neuron
    // current could then be extracted and used in iteration

    // CHANGE SO ASSIGNMENTS EVALUATED IN ORDER
    // for now have all eq assignments last (after change is applied)
    // or changes applied after consecutive diff eq assignments end
    // next set of changes applied when next set of diff eqs assigned

    // test creating default impl

    // allow 
    // on_spike: expr
    // and
    // on_spike:
    //     expr

    // default functions like max, min, exp, floor, ciel, heaviside
    // if function in same space as on iteration and on spike
    // add that function to the struct impl
    // function declarations in separate space from on iteration and on spike

    // refractory period (either if statements or separate block)

    // runge kutta and import integrators

    // check for syntax errors
    // could check to see if number of defintions matches
    // number of blocks ([item]-[end])
    // if not get each block and try individually parsing to see if
    // that returns the correct error

    // neuron def may need to be handled differently if voltage is not updated with dv/dt
    // if neuron is assigned with v =, similar to ion channels
    // or maybe in general, assignments should be done after changes calculated
    // but before changes applied
    // self.dw = self.w * self.dt
    // self.a = self.r
    // self.w += self.dw
    // or maybe option to do 
    // self.w += self.w * self.dt
    // instead
    // perhaps through a different integrator
    // or maybe all eq assignments before diff eq assignments

    // probably most elegant solution is something like the following
    // use --- to seperate iterations into seperate blocks
    // changes calculated at beginning of block and added at the end
    // to have something more sequential you can use blocking

    // handle function definitions in seperate block

    // handle ligand gates
    // neurotransmitter and approximate kinetics
    // handling spike trains
    // handling function if statements and boolean vars
    // handling plasticity

    let output_file_name = format!(
        "{}.rs", filename.as_str().split(".").collect::<Vec<&str>>()[0]
    );

    // collect import statements at the top
    // also collect code generated
    // stitch imports and code together and then write to file
    // imports will likely be a seperate struct that contains 
    // a field for neuron import and a field for ion channel imports
    // maybe add get_imports() method

    let iterate_and_spike_base = "use spiking_neural_networks::neuron::iterate_and_spike_traits::IterateAndSpikeBase;";
    let neuron_necessary_imports = vec![
        "CurrentVoltage", "GapConductance", "GaussianFactor", "LastFiringTime", "IsSpiking",
        "IterateAndSpike", "GaussianParameters", "LigandGatedChannels", 
        "Neurotransmitters", "NeurotransmitterKinetics", "ReceptorKinetics",
        "NeurotransmitterConcentrations",
    ];
    let neuron_necessary_imports = format!(
        "use spiking_neural_networks::neuron::iterate_and_spike::{{{}}};",
        neuron_necessary_imports.join(", ")
    );
    let neuron_necessary_imports = format!("{}\n{}", iterate_and_spike_base, neuron_necessary_imports);

    let mut imports = vec![];
    let mut code: HashMap<String, HashMap<String, String>> = HashMap::new();

    match ASTParser::parse(Rule::full, &contents) {
        Ok(pairs) => {
            for pair in pairs {
                match pair.as_rule() {
                    Rule::neuron_definition => {
                        let neuron_definition = generate_neuron(pair.into_inner())
                            .expect("Could not generate neuron");

                        let (neuron_imports, neuron_code) = neuron_definition.to_code();
    
                        if !imports.contains(&neuron_necessary_imports) {
                            imports.push(neuron_necessary_imports.clone());
                        }
                        if !imports.contains(&neuron_imports[0]) {
                            imports.push(neuron_imports[0].clone());
                        }
    
                        let neuron_type_name = neuron_definition.type_name.to_string();
    
                        let neuron_code_map = code.entry(String::from("neuron"))
                            .or_insert_with(HashMap::new);
                        
                        neuron_code_map.insert(neuron_type_name, neuron_code);
                    },
                    Rule::ion_channel_definition => {
                        let ion_channel = generate_ion_channel(pair.into_inner())
                            .expect("Could not generate ion channel");
    
                        let (ion_channel_imports, ion_channel_code) = ion_channel.to_code();
    
                        for i in ion_channel_imports {
                            if !imports.contains(&i) {
                                imports.push(i);
                            }
                        }
    
                        let ion_channel_type_name = ion_channel.type_name.to_string();
                        
                        let ion_channel_code_map = code.entry(String::from("ion_channel"))
                            .or_insert_with(HashMap::new);
    
                        ion_channel_code_map.insert(ion_channel_type_name, ion_channel_code);
                    },
                    _ => unreachable!("Unexpected definition: {:#?}", pair.as_rule()),
                }
            }
            
            // if any of the ion channel names found in neuron
            // (use substring to detect)
            // modify neuron code to insert proper update current code before dv changes

            // let ion_channel_names = code.get("ion_channel")
            //     .keys()
            //     .cloned()
            //     .collect();

            // let iteration_header = "fn iterate_and_spike(&mut self, input_current: f32) -> bool {";
            // let iteration_with_neurotransmission_start = "fn iterate_with_neurotransmitter_and_spike(";
            // let iteration_with_neurotransmission_args = vec![
            //     "&mut self", 
            //     "input_current: f32",
            //     "t_total: &NeurotransmitterConcentrations",
            // ];

            // code.get("neuron")
            //     .unwrap()
            //     .values_mut()
            //     .map(|i| {
            //         for i in ion_channel_names {
            //             if i.contains(&i) {
            //                 if code.get("ion_channel")
            //                     .unwrap()
            //                     .get(&i)
            //                     .contains("impl TimestepIndependentIonChannel") {
                                        // insert update current right after iterate and spike
                                        // header and iterate and spike with
                                        // neurotransmission header

                                        // use insert_at_substring method
            //                     }
            //             }
            //         }
            //     });

            let mut file = File::create(&output_file_name)?;
            file.write_all(imports.join("\n").as_bytes())?;
            file.write_all("\n\n\n".as_bytes())?;
            file.write_all(
                code.values()
                    .map(|i| i.values().map(|i| i.clone()).collect::<Vec<String>>().join("\n"))
                    .collect::<Vec<String>>()
                    .join("\n")
                    .as_bytes()
            )?;
        }
        Err(e) => {
            eprintln!("Parse failed: {:?}", e);
        }
    }

    Ok(())
}
