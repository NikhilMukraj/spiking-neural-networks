use std::collections::HashMap;
use std::fs::read_to_string;
use std::io::{Error, ErrorKind, Result};
use pest::Parser;
use pest::iterators::{Pair, Pairs};
use pest::error::{LineColLocation, ErrorVariant::{ParsingError, CustomError}};
use regex::Regex;

mod pest_ast;
use pest_ast::{ASTParser, Rule, PRATT_PARSER};

use syn::{LitStr, parse_macro_input};
use proc_macro::{
    Delimiter, Group, Ident, Literal, Punct, 
    Spacing, Span, TokenTree, TokenStream
};


#[derive(Debug)]
enum Ast {
    Number(f32),
    Name(String),
    UnaryMinus(Box<Ast>),
    NotOperator(Box<Ast>),
    BinOp {
        lhs: Box<Ast>,
        op: Op,
        rhs: Box<Ast>,
    },
    Function {
        name: String,
        args: Vec<Ast>
    },
    StructCall {
        name: String,
        attribute: String,
        args: Option<Vec<Ast>>,
    },
    StructFunctionCall {
        name: String,
        attribute: String,
        args: Vec<Ast>,
    },
    StructAssignment {
        name: String,
        type_name: String,
    },
    StructAssignments(Vec<Ast>),
    EqAssignment {
        name: String,
        expr: Box<Ast>,
    },
    DiffEqAssignment {
        name: String,
        expr: Box<Ast>,
    },
    FunctionAssignment {
        name: String,
        args: Vec<String>,
        expr: Box<Ast>,
    },
    TypeDefinition(String),
    OnSpike(Vec<Ast>),
    OnIteration(Vec<Ast>),
    SpikeDetection(Box<Ast>),
    GatingVariables(Vec<String>),
    VariableAssignment {
        name: String,
        value: Option<f32>,
    },
    VariablesAssignments(Vec<Ast>),
    IfStatement {
        condition: Box<Ast>,
        declarations: Vec<Ast>,
    }
}

#[derive(Debug)]
enum Op {
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

impl Ast {
    fn generate(&self) -> String {
        match self {
            Ast::Number(n) => {
                if n % 1. != 0. {
                    n.to_string()
                } else {
                    format!("{}.0", n)
                }
            },
            Ast::Name(name) => {
                if name == "v" {
                    String::from("self.current_voltage")
                } else if name == "i" {
                    String::from("input_current")
                } else {
                    format!("self.{}", name)
                }
            },
            Ast::UnaryMinus(expr) => format!("-{}", expr.generate()),
            Ast::NotOperator(expr) => format!("!{}", expr.generate()),
            Ast::BinOp { lhs, op, rhs } => {
                match op {
                    Op::Add => format!("({} + {})", lhs.generate(), rhs.generate()),
                    Op::Subtract => format!("({} - {})", lhs.generate(), rhs.generate()),
                    Op::Multiply => format!("({} * {})", lhs.generate(), rhs.generate()),
                    Op::Divide => format!("({} / {})", lhs.generate(), rhs.generate()),
                    Op::Power => format!("({}.powf({}))", lhs.generate(), rhs.generate()),
                    Op::Equal => format!("{} == {}", lhs.generate(), rhs.generate()),
                    Op::NotEqual => format!("{} != {}", lhs.generate(), rhs.generate()),
                    Op::GreaterThan => format!("{} > {}", lhs.generate(), rhs.generate()),
                    Op::GreaterThanOrEqual => format!("{} >= {}", lhs.generate(), rhs.generate()),
                    Op::LessThan => format!("{} < {}", lhs.generate(), rhs.generate()),
                    Op::LessThanOrEqual => format!("{} <= {}", lhs.generate(), rhs.generate()),
                    Op::And => format!("{} && {}", lhs.generate(), rhs.generate()),
                    Op::Or => format!("{} || {}", lhs.generate(), rhs.generate()),
                }
            }
            Ast::Function { name, args } => {
                format!(
                    "{}({})",
                    name, 
                    args.iter()
                        .map(|i| i.generate())
                        .collect::<Vec<String>>()
                        .join(", ")
                    )
            },
            Ast::StructCall { name, attribute, args } => {
                format!(
                    "self.{}.{}{}", 
                    name, 
                    attribute,
                    match args {
                        Some(args) => {
                            args.iter()
                                .map(|i| i.generate())
                                .collect::<Vec<String>>()
                                .join(", ")
                        },
                        None => String::from(""),
                    }
                )
            },
            Ast::StructFunctionCall { name, attribute, args } => {
                format!(
                    "self.{}.{}({});", 
                    name, 
                    attribute,
                    args.iter()
                        .map(|i| i.generate())
                        .collect::<Vec<String>>()
                        .join(", ")
                )
            },
            Ast::EqAssignment { name, expr } => {
                let name = if name == "v" {
                    String::from("self.current_voltage")
                } else {
                    format!("self.{}", name)
                };

                format!("{} = {};", name, expr.generate())
            },
            Ast::DiffEqAssignment { name, expr } => {
                format!("let d{} = ({}) * self.dt;", name, expr.generate())
            },
            Ast::FunctionAssignment{ name, args, expr } =>{
                format!(
                    "{}({}) = {}",
                    name, 
                    args.iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<String>>()
                        .join(", "),
                    expr.generate(),
                )
            },
            Ast::TypeDefinition(string) => string.clone(),
            Ast::OnSpike(assignments) => {
                assignments.iter()
                    .map(|i| i.generate())
                    .collect::<Vec<String>>()
                    .join("\n")
            },
            Ast::OnIteration(assignments) => {
                assignments.iter()
                    .map(|i| i.generate())
                    .collect::<Vec<String>>()
                    .join("\n\t\t")
            },
            Ast::SpikeDetection(expr) => { expr.generate() },
            Ast::GatingVariables(vars) => {
                format!("gating_vars: {}", vars.join(", "))
            },
            Ast::VariableAssignment { name, value } => {
                let value = match value {
                    Some(x) => x.to_string(),
                    None => String::from("None"),
                };

                format!("{} = {}", name, value)
            },
            Ast::VariablesAssignments(assignments) => {
                let assignments_string = assignments.iter()
                    .map(|i| i.generate())
                    .collect::<Vec<String>>()
                    .join("\n\t");

                format!("vars:\n\t{}", assignments_string)
            },
            Ast::StructAssignment { name, type_name } => {
                format!("{} = {}", name, type_name)
            },
            Ast::StructAssignments(assignments) => {
                let assignments_string = assignments.iter()
                    .map(|i| i.generate())
                    .collect::<Vec<String>>()
                    .join("\n\t");

                format!("structs:\n\t{}", assignments_string)
            },
            Ast::IfStatement { condition, declarations } => {
                format!(
                    "if {} {{\n{}\n}}", 
                    condition.generate(), 
                    declarations.iter()
                        .map(|i| i.generate())
                        .collect::<Vec<String>>()
                        .join("\n")
                )
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

struct NeuronDefinition {
    type_name: Ast,
    vars: Ast,
    on_spike: Option<Ast>,
    on_iteration: Ast,
    spike_detection: Ast,
    ion_channels: Option<Ast>,
}

const ITERATION_HEADER: &str = "fn iterate_and_spike(&mut self, input_current: f32) -> bool {";
const ITERATION_WITH_NEUROTRANSMITTER_START: &str = "fn iterate_with_neurotransmitter_and_spike(";
const ITERATION_WITH_NEUROTRANSMITTER_ARGS: [&str; 3] = [
    "&mut self", 
    "input_current: f32",
    "t_total: &NeurotransmitterConcentrations<Self::N>",
];

fn generate_iteration_with_neurotransmitter_header() -> String {
    format!(
        "{}\n\t{},\n) -> bool {{", 
        ITERATION_WITH_NEUROTRANSMITTER_START, 
        ITERATION_WITH_NEUROTRANSMITTER_ARGS.join(",\n\t"),
    )
}

fn generate_on_iteration(on_iteration: &Ast) -> String {
    let on_iteration_assignments = on_iteration.generate();

    let changes = match on_iteration {
        Ast::OnIteration(assignments) => {
            let mut assignments_strings = vec![];

            for i in assignments {
                if let Ast::DiffEqAssignment { name, .. } =  i {
                    let change_string = if name == "v" {
                        "self.current_voltage += dv;".to_string()
                    } else {
                        format!("self.{} += d{}", name, name)
                    };

                    assignments_strings.push(change_string);
                }
            }

            assignments_strings.join("\t\n")
        },
        _ => panic!("Expected on iteration AST")
    };

    format!("{}\n{}\n", on_iteration_assignments, changes)
}

fn generate_fields(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    format!("pub {}: f32", var_name)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
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
        let neurotransmitter_kind = "IonotropicNeurotransmitterType";

        let mut kinetics_import = vec![format!(
            "use spiking_neural_networks::neuron::iterate_and_spike::{{{}, {}}};",
            neurotransmitter_kinetics,
            receptor_kinetics,
        )];

        kinetics_import.push(
            format!(
                "use spiking_neural_networks::neuron::iterate_and_spike::{};",
                neurotransmitter_kind,
            )
        );

        let macros = "#[derive(Debug, Clone, IterateAndSpikeBase)]";
        let header = format!(
            "pub struct {}<T: NeurotransmitterKinetics, R: ReceptorKinetics> {{", 
            self.type_name.generate(),
        );

        let mut fields = generate_fields(&self.vars);

        let mut defaults = generate_defaults(&self.vars);

        let current_voltage_field = String::from("pub current_voltage: f32");
        let dt_field = String::from("pub dt: f32");
        let c_m_field = String::from("pub c_m: f32");
        let gap_conductance_field = String::from("pub gap_conductance: f32");
        let is_spiking_field = String::from("pub is_spiking: bool");
        let last_firing_time_field = String::from("pub last_firing_time: Option<usize>");
        let neurotransmitter_field = format!("pub synaptic_neurotransmitters: Neurotransmitters<{}, T>", neurotransmitter_kind);
        let receptors_field = String::from("pub receptors: LigandGatedChannels<R>");

        fields.insert(0, current_voltage_field);
        fields.push(gap_conductance_field);
        fields.push(dt_field);
        fields.push(c_m_field);

        let ion_channels = match &self.ion_channels {
            Some(Ast::StructAssignments(variables)) => {
                variables.iter()
                    .map(|i| {
                        let (var_name, type_name) = match i {
                            Ast::StructAssignment { name, type_name } => (name, type_name),
                            _ => unreachable!(),
                        };

                        format!("pub {}: {}", var_name, type_name)
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
        fields.push(neurotransmitter_field);
        fields.push(receptors_field);

        let fields = format!("\t{},", fields.join(",\n\t"));

        let impl_default = if !defaults.is_empty() {
            defaults.push(String::from("current_voltage: 0."));
            defaults.push(String::from("dt: 0.1"));
            defaults.push(String::from("c_m: 1."));
            defaults.push(String::from("gap_conductance: 10."));

            let default_ion_channels = match &self.ion_channels {
                Some(Ast::StructAssignments(variables)) => {
                    variables.iter()
                        .map(|i| {
                            let (var_name, type_name) = match i {
                                Ast::StructAssignment { name, type_name } => (name, type_name),
                                _ => unreachable!(),
                            };
    
                            format!("{}: {}::default()", var_name, type_name)
                        })
                        .collect::<Vec<String>>()
                },
                None => vec![],
                _ => unreachable!()
            };

            defaults.extend(default_ion_channels);
            defaults.push(String::from("is_spiking: false"));
            defaults.push(String::from("last_firing_time: None"));
            defaults.push(format!("synaptic_neurotransmitters: Neurotransmitters::<{}, T>::default()", neurotransmitter_kind));
            defaults.push(String::from("receptors: LigandGatedChannels::<R>::default()"));

            let default_fields = defaults.join(",\n\t");
            
            let default_function = format!(
                "fn default() -> Self {{ {} {{\n\t{}\n}}", 
                self.type_name.generate(),
                default_fields,
            );
            let default_function = add_indents(&default_function, "\t");

            format!(
                "\nimpl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for {}<T, R> {{\n\t{}\n}}\n}}\n",
                self.type_name.generate(),
                default_function,
            )
        } else {
            String::from("")
        };

        let handle_spiking_header = "fn handle_spiking(&mut self) -> bool {";

        let handle_spiking_function = match &self.on_spike {
            Some(value) => {
                let handle_spiking_check = "\tif self.is_spiking {";
                let handle_spiking_function = format!("\t\t{}", value.generate());

                format!("{}\n{}\n\t}}", handle_spiking_check, handle_spiking_function)
            },
            None => String::from(""),
        };

        let handle_spiking = if self.spike_detection.generate() != "continuous()" {
            let handle_is_spiking_calc = format!("\tself.is_spiking = {};", self.spike_detection.generate());
    
            format!(
                "{}\n{}\n{}\n\n\tself.is_spiking\n}}", 
                handle_spiking_header, 
                handle_is_spiking_calc,
                handle_spiking_function,
            )
        } else {
            let handle_is_spiking_calc = [
                "let increasing_right_now = last_voltage < self.current_voltage;",
                "let threshold_crossed = self.current_voltage > self.v_th;",
                "let is_spiking = threshold_crossed && self.was_increasing && !increasing_right_now;",
                "self.is_spiking = is_spiking;",
                "self.was_increasing = increasing_right_now;"
            ];
            let handle_is_spiking_calc = add_indents(&handle_is_spiking_calc.join("\n"), "\t");

            format!(
                "{}\n{}\n{}\n\tself.is_spiking\n}}",
                handle_spiking_header, 
                handle_is_spiking_calc,
                handle_spiking_function,
            )
        };

        let get_concentrations_header = "fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<Self::N> {";
        let get_concentrations_body = "self.synaptic_neurotransmitters.get_concentrations()";
        let get_concentrations_function = format!("{}\n\t{}\n}}", get_concentrations_header, get_concentrations_body);

        let handle_neurotransmitter_conc = "self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));";
        let handle_spiking_call = "self.handle_spiking()";
        let iteration_body = format!(
            "\n\t{}\n\t{}\n\t{}", 
            generate_on_iteration(&self.on_iteration), 
            handle_neurotransmitter_conc,
            handle_spiking_call,
        );
        let iteration_function = format!("{}{}\n}}", ITERATION_HEADER, iteration_body);

        let iteration_with_neurotransmitter_header = generate_iteration_with_neurotransmitter_header();

        let receptors_update = "self.receptors.update_receptor_kinetics(t_total, self.dt);";
        let receptors_set_current = "self.receptors.set_receptor_currents(self.current_voltage, self.dt);";

        let update_with_receptor_current = "self.current_voltage += self.receptors.get_receptor_currents(self.dt, self.c_m);";

        let iteration_with_neurotransmitter_body = format!(
            "\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}",
            receptors_update,
            receptors_set_current,
            generate_on_iteration(&self.on_iteration),
            update_with_receptor_current,
            handle_neurotransmitter_conc,
            handle_spiking_call,
        );

        let iteration_with_neurotransmitter_function = format!(
            "{}\n{}\n}}", 
            iteration_with_neurotransmitter_header,
            iteration_with_neurotransmitter_body,
        );

        let impl_header = format!(
            "impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> {}<T, R> {{", 
            self.type_name.generate()
        );
        let impl_body = add_indents(&handle_spiking, "\t");
        let impl_functions = format!("{}\n{}\n}}", impl_header, impl_body);

        let impl_header_iterate_and_spike = format!(
            "impl<T: NeurotransmitterKinetics, R: ReceptorKinetics> IterateAndSpike for {}<T, R> {{", 
            self.type_name.generate()
        );
        let impl_iterate_and_spike_body = format!(
            "type N = {};\n\n{}\n\n{}\n\n{}\n",
            neurotransmitter_kind,
            get_concentrations_function,
            iteration_function,
            iteration_with_neurotransmitter_function,
        );
        let impl_iterate_and_spike_body = add_indents(&impl_iterate_and_spike_body, "\t");
        let impl_iterate_and_spike = format!(
            "{}\n{}\n}}", 
            impl_header_iterate_and_spike, 
            impl_iterate_and_spike_body,
        );

        (
            kinetics_import,
            format!(
                "{}\n{}\n{}\n}}\n\n{}\n\n{}\n{}", 
                macros, 
                header, 
                fields, 
                impl_functions, 
                impl_iterate_and_spike,
                impl_default,
            )
        )
    }
}

fn parse_type_definition(pair: Pair<'_, Rule>) -> (String, Ast) {
    (
        String::from("type"), 
        Ast::TypeDefinition(
            String::from(pair.into_inner().next().unwrap().as_str())
        )
    )
}

fn parse_on_iteration(pair: Pair<'_, Rule>) -> (String, Ast) {
    let inner_rules = pair.into_inner();

    (
        String::from("on_iteration"),
        Ast::OnIteration(
            inner_rules
            .map(|i| parse_declaration(i))
            .collect::<Vec<Ast>>()
        )
    )
}

fn parse_on_spike(pair: Pair<'_, Rule>) -> (String, Ast) {
    let inner_rules = pair.into_inner();

    (
        String::from("on_spike"),
        Ast::OnSpike(
            inner_rules
            .map(|i| parse_declaration(i))
            .collect::<Vec<Ast>>()
        )
    )
}

fn parse_spike_detection(pair: Pair<'_, Rule>) -> (String, Ast) {
    (
        String::from("spike_detection"),
        Ast::SpikeDetection(Box::new(parse_bool_expr(pair.into_inner())))
    )
}

fn parse_vars_assignments(pair: Pair<'_, Rule>) -> (String, Ast) {
    let inner_rules = pair.into_inner();

    let assignments: Vec<Ast> = inner_rules
        .map(|i| Ast::VariableAssignment { 
            name: String::from(i.as_str()), 
            value: None,
        })
        .collect();

    (
        String::from("vars"),
        Ast::VariablesAssignments(assignments)
    )
}

fn parse_vars_with_default(pair: Pair<'_, Rule>) -> (String, Ast) {
    let inner_rules = pair.into_inner();

    let assignments: Vec<Ast> = inner_rules 
        .map(|i| {
            let mut nested_rule = i.into_inner();

            Ast::VariableAssignment { 
                name: String::from(nested_rule.next().unwrap().as_str()), 
                value: Some(
                    nested_rule.next()
                        .unwrap()
                        .as_str()
                        .parse::<f32>()
                        .unwrap()
                    ), 
            }
        })
        .collect(); 

    (
        String::from("vars"),
        Ast::VariablesAssignments(assignments)
    )
}

fn parse_ion_channels(pair: Pair<'_, Rule>) -> (String, Ast) {
    let inner_rules = pair.into_inner();

    let assignments: Vec<Ast> = inner_rules 
        .map(|i| {
            let mut nested_rule = i.into_inner();

            Ast::StructAssignment { 
                name: String::from(nested_rule.next().unwrap().as_str()), 
                type_name: String::from(
                    nested_rule.next()
                        .unwrap()
                        .as_str()
                )
            }
        })
        .collect(); 

    (
        String::from("ion_channels"),
        Ast::StructAssignments(assignments)
    )
}

fn parse_gating_variables(pair: Pair<'_, Rule>) -> (String, Ast) {
    let inner_rules = pair.into_inner();

    let assignments: Vec<String> = inner_rules 
        .map(|i| {
            String::from(i.as_str())
        })
        .collect(); 

    (
        String::from("gating_vars"),
        Ast::GatingVariables(assignments)
    )
}

fn generate_neuron(pairs: Pairs<Rule>) -> Result<NeuronDefinition> {
    let mut definitions: HashMap<String, Ast> = HashMap::new();

    for pair in pairs {
        let (key, current_ast) = match pair.as_rule() {
            Rule::type_def => {
                parse_type_definition(pair)
            },
            Rule::on_iteration_def => {
                parse_on_iteration(pair)
            },
            Rule::on_spike_def => {
                parse_on_spike(pair)
            },
            Rule::spike_detection_def => {
                parse_spike_detection(pair)
            }
            Rule::vars_def => {
                // if no defaults then just assume assingment is None
                // in order to prevent duplicate, key should be "vars"
                parse_vars_assignments(pair)
            },
            Rule::vars_with_default_def => {
                // assignment should be just a number
                // in order to prevent duplicate, key should be "vars"

                parse_vars_with_default(pair)
            },
            Rule::ion_channels_def => {
                parse_ion_channels(pair)
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

    let type_name = definitions.remove("type").ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "Type definition expected")
    })?;
    
    let vars = definitions.remove("vars").ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "Variables definition expected")
    })?;
    
    let spike_detection = definitions.remove("spike_detection").ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "Spike detection definition expected")
    })?;
    
    let on_iteration = definitions.remove("on_iteration").ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "On iteration definition expected")
    })?;
    
    let on_spike = definitions.remove("on_spike");
    let ion_channels = definitions.remove("ion_channels");

    Ok(
        NeuronDefinition {
            type_name,
            vars,
            spike_detection,
            on_iteration,
            on_spike,
            ion_channels,
        }
    )
}

fn generate_defaults(vars: &Ast) -> Vec<String> {
    let defaults = match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .filter_map(|i| {
                    if let Ast::VariableAssignment { name, value: Some(x) } = i {
                        if x % 1. == 0. {
                            Some(format!("{}: {}.", name, x))
                        } else {
                            Some(format!("{}: {}", name, x))
                        }
                    } else {
                        None
                    }
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!(),
    };

    defaults
}

struct IonChannelDefinition {
    type_name: Ast,
    vars: Ast,
    gating_vars: Option<Ast>,
    on_iteration: Ast,
}

impl IonChannelDefinition {
    fn get_use_timestep(&self) -> bool {
        match &self.on_iteration {
            Ast::OnIteration(assignments) => {
                let mut use_timestep = false;

                for i in assignments {
                    if let Ast::DiffEqAssignment { .. } = i {
                        use_timestep = true;
                    } else if let Ast::StructFunctionCall { args, .. } = i {
                        for arg in args {
                            if arg.generate() == "self.dt" {
                                use_timestep = true;
                            }
                        }
                    }
                }

                use_timestep
            },
            _ => unreachable!()
        }
    }

    fn to_code(&self) -> (Vec<String>, String) {
        let mut imports = vec![];

        let header = format!(
            "#[derive(Debug, Clone, Copy)]\npub struct {} {{", 
            self.type_name.generate(),
        );
        
        let mut fields = generate_fields(&self.vars);

        let gating_variables = match &self.gating_vars {
            Some(Ast::GatingVariables(variables)) => {
                imports.push(
                    String::from(
                        "use spiking_neural_networks::neuron::ion_channels::BasicGatingVariable;"
                    )
                );

                variables.clone()
                    .iter()
                    .map(|i| format!("pub {}: BasicGatingVariable", i))
                    .collect()
            },
            None => vec![],
            _ => unreachable!()
        };

        for i in gating_variables {
            fields.push(i)
        }

        let current_field = String::from("pub current: f32");
        fields.push(current_field);

        let fields = format!("\t{},", fields.join(",\n\t"));

        let use_timestep = self.get_use_timestep();

        let get_current = "fn get_current(&self) -> f32 { self.current }";

        let update_current = if use_timestep {
            let update_current_header = "fn update_current(&mut self, voltage: f32, dt: f32) {";

            // let mut lines: Vec<&str> = on_iteration.split('\n').collect();
            // let current_line_index = lines.iter().position(|&line| line.starts_with("self.current"));

            // let current_assignment = match current_line_index {
            //     Some(index) => lines.remove(index),
            //     None => "",
            // };

            // let update_current_body = add_indents(&lines.join("\n"), "\t");
            let update_current_body = generate_on_iteration(&self.on_iteration);

            let update_current_body = update_current_body.replace("self.current_voltage", "voltage");
            let update_current_body = update_current_body.replace("self.dt", "dt");

            format!(
                "{}\n{}\n}}", 
                update_current_header, 
                update_current_body,
            )
        } else {
            let update_current_header = "fn update_current(&mut self, voltage: f32) {";
            let update_current_body = add_indents(&self.on_iteration.generate(), "\t");
            let update_current_body = update_current_body.replace("self.current_voltage", "voltage");

            format!("{}\n{}\n}}", update_current_header, update_current_body)
        };
        
        // if use timestep then header is ionchannel
        // otherwise header is timestepindenpendentionchannel
        let impl_header = if use_timestep {
            format!("impl IonChannel for {} {{", self.type_name.generate())
        } else {
            format!("impl TimestepIndependentIonChannel for {} {{", self.type_name.generate())
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
        let get_current = add_indents(get_current, "\t");

        let mut defaults = generate_defaults(&self.vars);

        defaults.push(String::from("current: 0."));

        let gating_defaults = match &self.gating_vars {
            Some(Ast::GatingVariables(variables)) => {
                variables.clone()
                    .iter()
                    .map(|i| format!("{}: BasicGatingVariable::default()", i))
                    .collect()
            },
            None => vec![],
            _ => unreachable!()
        };

        defaults.extend(gating_defaults);

        let default_fields = defaults.join(",\n\t");
            
        let default_function = format!(
            "fn default() -> Self {{ {} {{\n\t{}\n}}", 
            self.type_name.generate(),
            default_fields,
        );
        let default_function = add_indents(&default_function, "\t");

        let default_function = format!(
            "\nimpl Default for {} {{\n\t{}\n}}\n}}\n",
            self.type_name.generate(),
            default_function,
        );
        let default_function = add_indents(&default_function, "\t");

        (
            imports, 
            format!(
                "{}\n{}\n}}\n\n{}\n\n{}\n{}\n\n{}\n}}\n", 
                header, 
                fields, 
                default_function,
                impl_header, 
                update_current, 
                get_current
            )
        )
    }
}

fn generate_ion_channel(pairs: Pairs<Rule>) -> Result<IonChannelDefinition> {
    let mut definitions: HashMap<String, Ast> = HashMap::new();

    for pair in pairs {
        let (key, current_ast) = match pair.as_rule() {
            Rule::type_def => {
                parse_type_definition(pair)
            },
            Rule::on_iteration_def => {
                parse_on_iteration(pair)
            },
            Rule::vars_def => {
                // if no defaults then just assume assingment is None
                // in order to prevent duplicate, key should be "vars"
                parse_vars_assignments(pair)
            },
            Rule::vars_with_default_def => {
                // assignment should be just a number
                // in order to prevent duplicate, key should be "vars"
                parse_vars_with_default(pair)
            },
            Rule::gating_variables_def => {
                parse_gating_variables(pair)
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

    let type_name = definitions.remove("type").ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "Type definition expected")
    })?;
    
    let vars = definitions.remove("vars").ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "Variables definition expected")
    })?;
    
    let gating_vars = definitions.remove("gating_vars");
    
    let on_iteration = definitions.remove("on_iteration").ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "On iteration definition expected")
    })?;
    
    Ok(
        IonChannelDefinition {
            type_name,
            vars,
            gating_vars,
            on_iteration,
        }
    )
}

struct NeurotransmitterKineticsDefinition {
    type_name: Ast,
    vars: Ast,
    on_iteration: Ast,
}

fn generate_neurotransmitter_kinetics(pairs: Pairs<Rule>) -> Result<NeurotransmitterKineticsDefinition> {
    let mut definitions: HashMap<String, Ast> = HashMap::new();

    for pair in pairs {
        let (key, current_ast) = match pair.as_rule() {
            Rule::type_def => {
                parse_type_definition(pair)
            },
            Rule::on_iteration_def => {
                parse_on_iteration(pair)
            },
            Rule::vars_with_default_def => {
                // assignment should be just a number
                // in order to prevent duplicate, key should be "vars"
                parse_vars_with_default(pair)
            },
            definition => unreachable!("Unexpected definition: {:#?}", definition)
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

    let type_name = definitions.remove("type").ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "Type name definition expected")
    })?;

    let vars = definitions.remove("vars").ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "Variables definition expected")
    })?;
    
    let on_iteration = definitions.remove("on_iteration").ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "On iteration definition expected")
    })?;
    
    Ok(NeurotransmitterKineticsDefinition { type_name, vars, on_iteration })    
}

impl NeurotransmitterKineticsDefinition {
    fn to_code(&self) -> (Vec<String>, String) {
        let imports = vec![String::from("NeurotransmitterKinetics")];

        let header = format!(
            "#[derive(Debug, Clone, Copy)]\npub struct {} {{", 
            self.type_name.generate(),
        );
        
        let mut fields = generate_fields(&self.vars);

        let t_field = String::from("pub t: f32");
        fields.push(t_field);

        let fields = format!("\t{},", fields.join(",\n\t"));

        let update_body = generate_on_iteration(&self.on_iteration);

        let mut defaults = generate_defaults(&self.vars);

        defaults.push(String::from("t: 0."));

        let default_fields = defaults.join(",\n\t");

        let default_function = format!(
            "fn default() -> Self {{ {} {{\n\t{}\n}}", 
            self.type_name.generate(),
            default_fields,
        );
        let default_function = add_indents(&default_function, "\t");

        let default_function = format!(
            "\nimpl Default for {} {{\n\t{}\n}}\n}}\n",
            self.type_name.generate(),
            default_function,
        );
        let default_function = add_indents(&default_function, "\t");

        let impl_header = format!("impl NeurotransmitterKinetics for {} {{", self.type_name.generate());

        let get_t = "fn(&self) -> f32 { self.t }";

        (
            imports,
            format!(
                "{}\n{}\n}}\n\n{}\n\n{}\n{}\n\n{}\n}}\n", 
                header, 
                fields, 
                default_function,
                impl_header, 
                update_body, 
                get_t
            )
        )
    }
}

// struct ReceptorKineticsDefinition {
    // vars: Ast,
    // on_iteration: Ast,
// }

fn parse_expr(pairs: Pairs<Rule>) -> Ast {
    PRATT_PARSER
        .map_primary(|primary| match primary.as_rule() {
            Rule::number => Ast::Number(primary.as_str().parse::<f32>().unwrap()),
            Rule::name => Ast::Name(String::from(primary.as_str())),
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

                let args: Option<Vec<Ast>> = inner_rules.next()
                    .map(|value| value.into_inner()
                        .map(|i| parse_expr(i.into_inner()))
                        .collect()
                    );
                
                Ast::StructCall { name, attribute, args }
            }
            Rule::function => {
                let mut inner_rules = primary.into_inner();

                let name: String = String::from(
                    inner_rules.next()
                        .expect("Could not get function name").as_str()
                );

                let args: Vec<Ast> = inner_rules.next()
                    .expect("No arguments found")
                    .into_inner()
                    .map(|i| parse_expr(i.into_inner()))
                    .collect();
                
                Ast::Function { name, args }
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
            Ast::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
            }
        })
        .map_prefix(|op, rhs| match op.as_rule() {
            Rule::unary_minus => Ast::UnaryMinus(Box::new(rhs)),
            _ => unreachable!(),
        })
        .parse(pairs)
}

fn parse_bool_expr(pairs: Pairs<Rule>) -> Ast {
    PRATT_PARSER
        .map_primary(|primary| match primary.as_rule() {
            Rule::number => Ast::Number(primary.as_str().parse::<f32>().unwrap()),
            Rule::name => Ast::Name(String::from(primary.as_str())),
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

                let args: Option<Vec<Ast>> = inner_rules.next()
                    .map(|value| value.into_inner()
                        .map(|i| parse_bool_expr(i.into_inner()))
                        .collect()
                    );
                
                Ast::StructCall { name, attribute, args }
            },
            Rule::function => {
                let mut inner_rules = primary.into_inner();

                let name: String = String::from(inner_rules.next()
                    .expect("Could not get function name").as_str()
                );

                let args: Vec<Ast> = inner_rules.next()
                    .expect("No arguments found")
                    .into_inner()
                    .map(|i| parse_bool_expr(i.into_inner()))
                    .collect();
                
                Ast::Function { name, args }
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
            Ast::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
            }
        })
        .map_prefix(|op, rhs| match op.as_rule() {
            Rule::unary_minus => Ast::UnaryMinus(Box::new(rhs)),
            Rule::not_operator => Ast::NotOperator(Box::new(rhs)),
            _ => unreachable!(),
        })
        .parse(pairs)
}

fn parse_declaration(pair: Pair<Rule>) -> Ast {
    match pair.as_rule() {
        Rule::struct_call_execution => {
            let mut inner_rules = pair.into_inner();

            let name: String = String::from(inner_rules.next()
                .expect("Could not get function name").as_str()
            );

            let attribute: String = String::from(inner_rules.next()
                .expect("Could not get attribute name").as_str()
            );

            let args: Vec<Ast> = inner_rules.next()
                .unwrap()
                .into_inner()
                .map(|i| parse_expr(i.into_inner()))
                .collect();

            Ast::StructFunctionCall { name, attribute, args }
        },
        Rule::diff_eq_declaration => {
            let mut inner_rules = pair.into_inner();

            let name: String = String::from(inner_rules.next()
                .expect("Could not get function name").as_str()
            );

            let expr: Box<Ast> = Box::new(
                parse_expr(
                    inner_rules.next()
                        .expect("No arguments found")
                        .into_inner()
                )
            );

            Ast::DiffEqAssignment { name, expr }
        },
        Rule::eq_declaration => {
            let mut inner_rules = pair.into_inner();

            let name: String = String::from(inner_rules.next()
                .expect("Could not get function name").as_str()
            );

            let expr: Box<Ast> = Box::new(
                parse_expr(
                    inner_rules.next()
                        .expect("No arguments found")
                        .into_inner()
                )
            );

            Ast::EqAssignment { name, expr }
        },
        Rule::func_declaration => {
            let mut inner_rules = pair.into_inner();
            let name = String::from(inner_rules.next().unwrap().as_str());

            let args = inner_rules.next().unwrap()
                .into_inner()
                .map(|arg| String::from(arg.as_str()))
                .collect::<Vec<String>>();

            let expr = Box::new(parse_expr(inner_rules.next().unwrap().into_inner()));

            Ast::FunctionAssignment {
                name,
                args,
                expr,
            }
        },
        Rule::if_statement => {
            let mut inner_rules = pair.into_inner();

            let condition = Box::new(parse_bool_expr(inner_rules.next().unwrap().into_inner()));

            let declarations: Vec<Ast> = inner_rules
                .map(|i| parse_declaration(i))
                .collect();

            Ast::IfStatement { condition, declarations }
        }
        rule => unreachable!("Unexpected declaration, found {:#?}", rule),
    }
}

fn extract_name_from_pattern(string: &str, i: &str) -> Vec<String> {
    let re = Regex::new(&format!(r"pub (.*): {}", i)).unwrap();
    let mut output = vec![];

    for caps in re.captures_iter(string) {
        let first_part = &caps[1];
        if string.contains(i) {
            output.push(first_part.to_string());
        }
    }

    output
}

fn insert_at_substring(original: &str, to_find: &str, to_insert: &str) -> String {
    if let Some(start) = original.find(to_find) {
        let mut result = String::new();
        result.push_str(&original[..start + to_find.len()]);
        result.push_str(to_insert);
        result.push_str(&original[start + to_find.len()..]);

        result
    } else {
        String::from(original)
    }
}

fn build_function(model_description: String) -> TokenStream {
    let iterate_and_spike_base = "use spiking_neural_networks::neuron::iterate_and_spike_traits::IterateAndSpikeBase;";
    let neuron_necessary_imports = [
        "CurrentVoltage", "GapConductance", "LastFiringTime", "IsSpiking",
        "Timestep", "IterateAndSpike", "LigandGatedChannels", 
        "Neurotransmitters", "NeurotransmitterKinetics", "ReceptorKinetics",
        "NeurotransmitterConcentrations"
    ];
    let neuron_necessary_imports = format!(
        "
        use spiking_neural_networks::neuron::intermediate_delegate::NeurotransmittersIntermediate;\n
        use spiking_neural_networks::neuron::iterate_and_spike::{{{}}};
        ",
        neuron_necessary_imports.join(", ")
    );
    let neuron_necessary_imports = format!("{}\n{}", iterate_and_spike_base, neuron_necessary_imports);
    
    let mut imports = vec![];
    let mut code: HashMap<String, HashMap<String, String>> = HashMap::new();
    
    match ASTParser::parse(Rule::full, &model_description) {
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
                        for i in &neuron_imports {
                            if !imports.contains(i) {
                                imports.push(i.clone());
                            }
                        }
    
                        let neuron_type_name = neuron_definition.type_name.generate();
    
                        let neuron_code_map = code.entry(String::from("neuron"))
                            .or_default();
                    
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
    
                        let ion_channel_type_name = ion_channel.type_name.generate();
                    
                        let ion_channel_code_map = code.entry(String::from("ion_channel"))
                            .or_default();
    
                        ion_channel_code_map.insert(ion_channel_type_name, ion_channel_code);
                    },
                    Rule::neurotransmitter_kinetics_definition => {
                        let neurotransmitter_kinetics = generate_neurotransmitter_kinetics(pair.into_inner())
                            .expect("Could not generate neurotransmitter kinetics");

                        let (neurotransmitter_kinetics_imports, neurotransmitter_kinetics_code) = neurotransmitter_kinetics.to_code();

                        for i in neurotransmitter_kinetics_imports {
                            if !imports.contains(&i) {
                                imports.push(i);
                            }
                        }
    
                        let neurotransmitter_kinetics_type_name = neurotransmitter_kinetics.type_name.generate();
                    
                        let neurotransmitter_kinetics_code_map = code.entry(String::from("neurotransmitter_kinetics"))
                            .or_default();
    
                        neurotransmitter_kinetics_code_map.insert(
                            neurotransmitter_kinetics_type_name, neurotransmitter_kinetics_code
                        );
                    },
                    Rule::receptor_kinetics_definition => {
                        todo!()
                    },
                    Rule::EOI => {
                        continue
                    }
                    _ => unreachable!("Unexpected definition: {:#?}", pair.as_rule()),
                }
            }
        
            // if any of the ion channel names found in neuron
            // (use substring to detect)
            // modify neuron code to insert proper update current code before dv changes
    
            let ion_channel_data: HashMap<String, bool> = code.get("ion_channel")
                .unwrap_or(&HashMap::<String, String>::new())
                .iter()
                .map(|(name, code)| {
                    let is_timestep_independent = code.contains("impl TimestepIndependentIonChannel");
                    (name.clone(), is_timestep_independent)
                })
                .collect();
    
            let iteration_with_neurotransmitter_header = add_indents(
                &generate_iteration_with_neurotransmitter_header(), "\t"
            );
    
            if let Some(neuron_code_map) = code.get_mut("neuron") {
                for i in neuron_code_map.values_mut() {
                    for (ion_channel_name, is_timestep_independent) in &ion_channel_data {
                        if i.contains(ion_channel_name) {
                            let names = extract_name_from_pattern(i, ion_channel_name);
    
                            for name in names {
                                let to_insert = if *is_timestep_independent {
                                    format!(
                                        "\n\t\tself.{}.update_current(self.current_voltage, self.dt);",
                                        name
                                    )
                                } else {
                                    format!(
                                        "\n\t\tself.{}.update_current(self.current_voltage);",
                                        name
                                    )
                                };
    
                                *i = insert_at_substring(
                                    i, 
                                    ITERATION_HEADER,
                                    &to_insert,
                                );
    
                                *i = insert_at_substring(
                                    i, 
                                    &iteration_with_neurotransmitter_header,
                                    &to_insert,
                                );
                            }
                        }
                    }
                }
            }
    
            let mut functions: HashMap<String, String> = HashMap::new();
            functions.insert(
                String::from("max"), 
                String::from("fn max(a: f32, b: f32) -> f32 { a.max(b) }"),
            );
            functions.insert(
                String::from("min"),
                String::from("fn min(a: f32, b: f32) -> f32 { a.min(b) }")
            );
            functions.insert(
                String::from("exp"),
                String::from("fn exp(x: f32) -> f32 { x.exp() }") 
            );
            functions.insert(
                String::from("tanh"),
                String::from("fn tanh(x: f32) -> f32 { x.tanh() }"),
            );
            functions.insert(
                String::from("sinh"),
                String::from("fn sinh(x: f32) -> f32 { x.sinh() }"),
            );
            functions.insert(
                String::from("cosh"),
                String::from("fn cosh(x: f32) -> f32 { x.cosh() }"),
            );
            functions.insert(
                String::from("tan"),
                String::from("fn tan(x: f32) -> f32 { x.tan() }"),
            );
            functions.insert(
                String::from("sin"),
                String::from("fn sin(x: f32) -> f32 { x.sin() }"),
            );
            functions.insert(
                String::from("cos"),
                String::from("fn cos(x: f32) -> f32 { x.cos() }"),
            );
            functions.insert(
                String::from("heaviside"),
                String::from("fn heaviside(x: f32) -> f32 { if x < 0 { 0 } else { x }"),
            );
            // continous is also a reserved function name
    
            let mut functions_to_add = Vec::new();
    
            let mut all_code = code.values()
                .map(|i| i.values().cloned().collect::<Vec<String>>().join("\n"))
                .collect::<Vec<String>>()
                .join("\n");
    
            // check whitespace or ( before function to ensure that it is a function call
            // (do not need to check for operator) as long as code generator formats with whitespace
            for (key, value) in functions.iter() {
                if all_code.contains(&format!(" {}(", key)) || all_code.contains(&format!("({}(", key)) {
                    functions_to_add.push(value.clone());
                }
            }
    
            all_code = if !functions_to_add.is_empty() {
                format!("{}\n{}\n", all_code, functions_to_add.join("\n\n"))
            } else {
                all_code
            };
    
            // println!("{}\n\n\n{}", imports.join("\n"), all_code);
    
            format!("{}\n\n\n{}", imports.join("\n"), all_code)
                .parse::<TokenStream>().unwrap()
        }
        Err(e) => {
            let mut error_out = format!("Parse failed: {:?}", e);
    
            match e.line_col {
                // Handle the case where the error is at a single position
                LineColLocation::Pos((line_number, _)) => {
                    let lines: Vec<&str> = model_description.lines().collect();
                    if line_number > 0 && line_number <= lines.len() {
                        error_out = format!("{}, Error occurred at line {}: {}", error_out, line_number, lines[line_number - 1]);
                    } else if line_number == lines.len() + 1 {
                        error_out = format!("{}, Error occured at line: {}", error_out, line_number);
                    } else {
                        error_out = format!("{}, Line number {} is out of bounds", error_out, line_number);
                    }
                }
                // Handle the case where the error spans multiple positions
                LineColLocation::Span((start_line, _), (end_line, _)) => {
                    let lines: Vec<&str> = model_description.lines().collect();
                    if start_line > 0 && start_line <= lines.len() && end_line > 0 && end_line <= lines.len() {
                        error_out = format!("{}, Error starts at line {}: {}", error_out, start_line, lines[start_line - 1]);
                        if start_line != end_line {
                            error_out = format!("{}, Error ends at line {}: {}", error_out, end_line, lines[end_line - 1]);
                        }
                    } else if start_line > 0 && start_line <= lines.len() && end_line > 0 && end_line == lines.len() + 1 { 
                        error_out = format!("{}, Error starts at line {}: {}", error_out, start_line, lines[start_line - 1]);
                        if start_line != end_line {
                            error_out = format!("{}, Error ends at line {}", error_out, end_line);
                        }
                    } else {
                        error_out = format!("{}, Line numbers are out of bounds", error_out);
                    }
                }
            }
    
            match &e.variant {
                ParsingError { positives, negatives } => {
                    if !positives.is_empty() {
                        error_out = format!("{}, Expected to find: {:?}", error_out, positives);
                    }
                    if !negatives.is_empty() {
                        error_out = format!("{}, Did not expect to find: {:?}", error_out, negatives);
                    }
                }
                CustomError { message } => {
                    error_out = format!("{}, Custom error: {}", error_out, message);
                }
            }
    
            [
                TokenTree::Ident(Ident::new("compile_error", Span::mixed_site())),
                TokenTree::Punct(Punct::new('!', Spacing::Alone)),
                TokenTree::Group(Group::new(
                    Delimiter::Parenthesis,
                    [TokenTree::Literal(Literal::string(&error_out))].into_iter().collect(),
                )),
            ]
            .into_iter()
            .collect()
        }
    }
}

#[proc_macro]
pub fn neuron_builder(model_description: TokenStream) -> TokenStream {
    // block based seperation

    // test creating default impl
    // default values for build in values, (gap conductance, dt, v)

    // handle comments by stripping comments with regex

    // function declarations in separate space from on iteration and on spike
    
    // custom ligand gates implementation given a new neurotransmitter type set

    // refractory period (either if statements or separate block)

    // runge kutta and import integrators

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

    // collect import statements at the top
    // also collect code generated
    // stitch imports and code together and then write to file
    // imports will likely be a seperate struct that contains 
    // a field for neuron import and a field for ion channel imports
    // maybe add get_imports() method

    let model_description = parse_macro_input!(model_description as LitStr);
    let model_description = model_description.value();

    build_function(model_description)
}

#[proc_macro]
pub fn neuron_builder_from_file(filename: TokenStream) -> TokenStream {
    let filename = parse_macro_input!(filename as LitStr);
    let filename = filename.value();

    build_function(read_to_string(filename).expect("Could not read file to string"))
}
