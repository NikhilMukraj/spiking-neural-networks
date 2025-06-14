use std::collections::HashMap;
use std::fs::read_to_string;
use std::io::{Error, ErrorKind, Result};
use pest::Parser;
use pest::iterators::{Pair, Pairs};
use pest::error::{LineColLocation, ErrorVariant::{ParsingError, CustomError}};
// use regex::Regex;

mod pest_ast;
use pest_ast::{ASTParser, Rule, PRATT_PARSER};

use syn::{LitStr, parse_macro_input};
use proc_macro::{
    Delimiter, Group, Ident, Literal, Punct, 
    Spacing, Span, TokenTree, TokenStream
};


#[derive(Debug, Clone, Copy)]
enum NumOrBool {
    Number(f32),
    Bool(bool),
}

#[derive(Debug, Clone)]
enum Ast {
    Number(f32),
    Bool(bool),
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
        eq_operator: String,
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
    KineticsDefinition(String, String),
    SingleKineticsDefinition(String),
    OnSpike(Vec<Ast>),
    OnIteration(Vec<Ast>),
    SpikeDetection(Box<Ast>),
    GatingVariables(Vec<String>),
    VariableAssignment {
        name: String,
        value: NumOrBool,
    },
    VariablesAssignments(Vec<Ast>),
    IfStatement {
        conditions: Vec<Ast>,
        declarations: Vec<Vec<Ast>>,
    }
}

#[derive(Debug, Clone, Copy)]
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
            Ast::Bool(val) => val.to_string(),
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
                match args {
                    Some(args) => {
                        format!(
                            "self.{}.{}({})", 
                            name, 
                            attribute,
                            args.iter()
                                .map(|i| i.generate())
                                .collect::<Vec<String>>()
                                .join(", ")
                        )
                    },
                    None => {
                        format!("self.{}.{}", name, attribute)
                    }
                }
            }
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
            Ast::EqAssignment { name, eq_operator, expr } => {
                let name = if name == "v" {
                    String::from("self.current_voltage")
                } else {
                    format!("self.{}", name)
                };

                format!("{} {} {};", name, eq_operator, expr.generate())
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
            Ast::KineticsDefinition(neuro, receptor) => format!("{}, {}", neuro, receptor),
            Ast::SingleKineticsDefinition(string) => string.clone(),
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
                let value_str = match value {
                    NumOrBool::Number(x) => x.to_string(),
                    NumOrBool::Bool(x) => x.to_string(),
                };

                format!("{} = {}", name, value_str)
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
            Ast::IfStatement { conditions, declarations } => {
                if conditions.len() == 1 && declarations.len() == 1 {
                    format!(
                        "if {} {{\n{}\n}}", 
                        conditions[0].generate(), 
                        declarations[0].iter()
                            .map(|i| i.generate())
                            .collect::<Vec<String>>()
                            .join("\n")
                    )
                } else if conditions.len() == 1 {
                    format!(
                        "if {} {{\n{}\n}} else {{\n{}\n}}",
                        conditions[0].generate(), 
                        declarations[0].iter()
                            .map(|i| i.generate())
                            .collect::<Vec<String>>()
                            .join("\n"),
                        declarations[1].iter()
                            .map(|i| i.generate())
                            .collect::<Vec<String>>()
                            .join("\n"),
                    )
                } else {
                    let mut result = String::new();

                    result.push_str(&format!(
                        "if {} {{\n{}\n}}", 
                        conditions[0].generate(), 
                        declarations[0].iter()
                            .map(|i| i.generate())
                            .collect::<Vec<String>>()
                            .join("\n")
                    ));

                    for i in 1..conditions.len() {
                        result.push_str(&format!(
                            " else if {} {{\n{}\n}}", 
                            conditions[i].generate(), 
                            declarations[i].iter()
                                .map(|i| i.generate())
                                .collect::<Vec<String>>()
                                .join("\n")
                        ));
                    }

                    if conditions.len() != declarations.len() {
                        result.push_str(&format!(
                            " else {{\n{}\n}}", 
                            declarations[declarations.len() - 1].iter()
                                .map(|i| i.generate())
                                .collect::<Vec<String>>()
                                .join("\n")
                        ));
                    }

                    result
                }
            }
        }
    }

    #[cfg(feature="gpu")] 
    fn generate_non_kernel_gpu(&self) -> String {
        match &self {
            Ast::Number(n) => {
                if n % 1. != 0. {
                    format!("{}f", n)
                } else {
                    format!("{}.0f", n)
                }
            },
            Ast::Bool(val) => val.to_string(),
            Ast::Name(name) => {
                if name == "v" {
                    String::from("current_voltage")
                } else if name == "i" {
                    String::from("input_current")
                } else {
                    name.to_string()
                }
            },
            Ast::UnaryMinus(expr) => format!("-{}", expr.generate_non_kernel_gpu()),
            Ast::NotOperator(expr) => format!("!{}", expr.generate_non_kernel_gpu()),
            Ast::BinOp { lhs, op, rhs } => {
                match op {
                    Op::Add => format!("({} + {})", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                    Op::Subtract => format!("({} - {})", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                    Op::Multiply => format!("({} * {})", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                    Op::Divide => format!("({} / {})", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                    Op::Power => format!("pow({}, {}))", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                    Op::Equal => format!("{} == {}", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                    Op::NotEqual => format!("{} != {}", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                    Op::GreaterThan => format!("{} > {}", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                    Op::GreaterThanOrEqual => format!("{} >= {}", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                    Op::LessThan => format!("{} < {}", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                    Op::LessThanOrEqual => format!("{} <= {}", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                    Op::And => format!("{} && {}", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                    Op::Or => format!("{} || {}", lhs.generate_non_kernel_gpu(), rhs.generate_non_kernel_gpu()),
                }
            }
            Ast::Function { name, args } => {
                format!(
                    "{}({})",
                    name, 
                    args.iter()
                        .map(|i| i.generate_non_kernel_gpu())
                        .collect::<Vec<String>>()
                        .join(", ")
                    )
            },
            Ast::StructCall { name, attribute, args } => {
                match args {
                    Some(args) => {
                        format!(
                            "{}${}({})", 
                            name, 
                            attribute,
                            args.iter()
                                .map(|i| i.generate_non_kernel_gpu())
                                .collect::<Vec<String>>()
                                .join(", ")
                        )
                    },
                    None => {
                        format!("{}${}", name, attribute)
                    }
                }
            }
            Ast::EqAssignment { name, eq_operator, expr } => {
                let name = if name == "v" {
                    String::from("current_voltage")
                } else {
                    name.to_string()
                };

                format!("{} {} {};", name, eq_operator, expr.generate_non_kernel_gpu())
            },
            Ast::DiffEqAssignment { name, expr } => {
                format!("d{} = ({}) * dt;", name, expr.generate_non_kernel_gpu())
            },
            Ast::OnIteration(assignments) => {
                assignments.iter()
                    .map(|i| i.generate_non_kernel_gpu())
                    .collect::<Vec<String>>()
                    .join("\n\t\t")
            },
            Ast::IfStatement { conditions, declarations } => {
                if conditions.len() == 1 && declarations.len() == 1 {
                    format!(
                        "if ({}) {{\n{}\n}}", 
                        conditions[0].generate_non_kernel_gpu(), 
                        declarations[0].iter()
                            .map(|i| i.generate_non_kernel_gpu())
                            .collect::<Vec<String>>()
                            .join("\n")
                    )
                } else if conditions.len() == 1 {
                    format!(
                        "if ({}) {{\n{}\n}} else {{\n{}\n}}",
                        conditions[0].generate_non_kernel_gpu(), 
                        declarations[0].iter()
                            .map(|i| i.generate_non_kernel_gpu())
                            .collect::<Vec<String>>()
                            .join("\n"),
                        declarations[1].iter()
                            .map(|i| i.generate_non_kernel_gpu())
                            .collect::<Vec<String>>()
                            .join("\n"),
                    )
                } else {
                    let mut result = String::new();

                    result.push_str(&format!(
                        "if ({}) {{\n{}\n}}", 
                        conditions[0].generate_non_kernel_gpu(), 
                        declarations[0].iter()
                            .map(|i| i.generate_non_kernel_gpu())
                            .collect::<Vec<String>>()
                            .join("\n")
                    ));

                    for i in 1..conditions.len() {
                        result.push_str(&format!(
                            " else if ({}) {{\n{}\n}}", 
                            conditions[i].generate_non_kernel_gpu(), 
                            declarations[i].iter()
                                .map(|i| i.generate_non_kernel_gpu())
                                .collect::<Vec<String>>()
                                .join("\n")
                        ));
                    }

                    if conditions.len() != declarations.len() {
                        result.push_str(&format!(
                            " else {{\n{}\n}}", 
                            declarations[declarations.len() - 1].iter()
                                .map(|i| i.generate_non_kernel_gpu())
                                .collect::<Vec<String>>()
                                .join("\n")
                        ));
                    }

                    result
                }
            },
            ast => panic!("Non kernel GPU code for {:#?} is not implemented", ast)
        }
    }

    #[cfg(feature = "gpu")]
    fn generate_kernel_gpu(&self) -> String {
        match self {
            Ast::Number(n) => {
                if n % 1. != 0. {
                    format!("{}f", n)
                } else {
                    format!("{}.0f", n)
                }
            },
            Ast::Bool(val) => val.to_string(),
            Ast::Name(name) => {
                if name == "v" {
                    String::from("current_voltage[index]")
                } else if name == "i" {
                    String::from("inputs[index]")
                } else {
                    format!("{}[index]", name)
                }
            },
            Ast::UnaryMinus(expr) => format!("-{}", expr.generate_kernel_gpu()),
            Ast::NotOperator(expr) => format!("!{}", expr.generate_kernel_gpu()),
            Ast::BinOp { lhs, op, rhs } => {
                match op {
                    Op::Add => format!("({} + {})", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                    Op::Subtract => format!("({} - {})", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                    Op::Multiply => format!("({} * {})", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                    Op::Divide => format!("({} / {})", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                    Op::Power => format!("(pow({}, {}))", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                    Op::Equal => format!("{} == {}", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                    Op::NotEqual => format!("{} != {}", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                    Op::GreaterThan => format!("{} > {}", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                    Op::GreaterThanOrEqual => format!("{} >= {}", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                    Op::LessThan => format!("{} < {}", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                    Op::LessThanOrEqual => format!("{} <= {}", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                    Op::And => format!("{} && {}", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                    Op::Or => format!("{} || {}", lhs.generate_kernel_gpu(), rhs.generate_kernel_gpu()),
                }
            }
            Ast::Function { name, args } => {
                format!(
                    "{}({})",
                    name, 
                    args.iter()
                        .map(|i| i.generate_kernel_gpu())
                        .collect::<Vec<String>>()
                        .join(", ")
                    )
            },
            Ast::StructCall { name, attribute, args } => {
                match args {
                    Some(args) => {
                        format!(
                            "{}${}({})", 
                            name, 
                            attribute,
                            args.iter()
                                .map(|i| i.generate_kernel_gpu())
                                .collect::<Vec<String>>()
                                .join(", ")
                        )
                    },
                    None => {
                        format!("{}${}", name, attribute)
                    }
                }
            }
            Ast::StructFunctionCall { name, attribute, args } => {
                format!(
                    "{}${}({});", 
                    name, 
                    attribute,
                    args.iter()
                        .map(|i| i.generate_kernel_gpu())
                        .collect::<Vec<String>>()
                        .join(", ")
                )
            },
            Ast::EqAssignment { name, eq_operator, expr } => {
                let name = if name == "v" {
                    String::from("current_voltage[index]")
                } else {
                    format!("{}[index]", name)
                };

                format!("{} {} {};", name, eq_operator, expr.generate_kernel_gpu())
            },
            Ast::DiffEqAssignment { name, expr } => {
                format!("float d{} = ({}) * dt[index];", name, expr.generate_kernel_gpu())
            },
            Ast::TypeDefinition(string) => string.clone(),
            Ast::KineticsDefinition(neuro, receptor) => format!("{}, {}", neuro, receptor),
            Ast::SingleKineticsDefinition(string) => string.clone(),
            Ast::OnSpike(assignments) => {
                assignments.iter()
                    .map(|i| i.generate_kernel_gpu())
                    .collect::<Vec<String>>()
                    .join("\n")
            },
            Ast::OnIteration(assignments) => {
                assignments.iter()
                    .map(|i| i.generate_kernel_gpu())
                    .collect::<Vec<String>>()
                    .join("\n\t\t")
            },
            Ast::SpikeDetection(expr) => { expr.generate_kernel_gpu() },
            Ast::VariableAssignment { name, value } => {
                let value_str = match value {
                    NumOrBool::Number(x) => x.to_string(),
                    NumOrBool::Bool(x) => x.to_string(),
                };

                format!("{} = {}", name, value_str)
            },
            Ast::IfStatement { conditions, declarations } => {
                if conditions.len() == 1 && declarations.len() == 1 {
                    format!(
                        "if ({}) {{\n{}\n}}", 
                        conditions[0].generate_kernel_gpu(), 
                        declarations[0].iter()
                            .map(|i| i.generate_kernel_gpu())
                            .collect::<Vec<String>>()
                            .join("\n")
                    )
                } else if conditions.len() == 1 {
                    format!(
                        "if ({}) {{\n{}\n}} else {{\n{}\n}}",
                        conditions[0].generate_kernel_gpu(), 
                        declarations[0].iter()
                            .map(|i| i.generate_kernel_gpu())
                            .collect::<Vec<String>>()
                            .join("\n"),
                        declarations[1].iter()
                            .map(|i| i.generate_kernel_gpu())
                            .collect::<Vec<String>>()
                            .join("\n"),
                    )
                } else {
                    let mut result = String::new();

                    result.push_str(&format!(
                        "if ({}) {{\n{}\n}}", 
                        conditions[0].generate_kernel_gpu(), 
                        declarations[0].iter()
                            .map(|i| i.generate_kernel_gpu())
                            .collect::<Vec<String>>()
                            .join("\n")
                    ));

                    for i in 1..conditions.len() {
                        result.push_str(&format!(
                            " else if ({}) {{\n{}\n}}", 
                            conditions[i].generate_kernel_gpu(), 
                            declarations[i].iter()
                                .map(|i| i.generate_kernel_gpu())
                                .collect::<Vec<String>>()
                                .join("\n")
                        ));
                    }

                    if conditions.len() != declarations.len() {
                        result.push_str(&format!(
                            " else {{\n{}\n}}", 
                            declarations[declarations.len() - 1].iter()
                                .map(|i| i.generate_kernel_gpu())
                                .collect::<Vec<String>>()
                                .join("\n")
                        ));
                    }

                    result
                }
            },
            ast => panic!("{:#?} is unimplemented for GPU kernel", ast),
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
    kinetics: Option<Ast>,
    vars: Ast,
    on_spike: Option<Ast>,
    on_iteration: Ast,
    spike_detection: Ast,
    ion_channels: Option<Ast>,
    on_electrochemical_iteration: Option<Ast>,
    receptors: Option<Ast>,
}

const ITERATION_HEADER: &str = "fn iterate_and_spike(&mut self, input_current: f32) -> bool {";
const ITERATION_WITH_NEUROTRANSMITTER_START: &str = "fn iterate_with_neurotransmitter_and_spike(";
const ITERATION_WITH_NEUROTRANSMITTER_ARGS: [&str; 3] = [
    "&mut self", 
    "input_current: f32",
    "t: &NeurotransmitterConcentrations<Self::N>",
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
                        format!("self.{} += d{};", name, name)
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

fn generate_fields_internal<F: Fn(&str, &str) -> String>(vars: &Ast, format_type: F) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    let type_name = match i {
                        Ast::VariableAssignment { value, .. } => match value {
                            NumOrBool::Number(_) => "f32",
                            NumOrBool::Bool(_) => "bool",
                        },
                        _ => unreachable!(),
                    };

                    format_type(var_name, type_name)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

fn generate_fields(vars: &Ast) -> Vec<String> {
    generate_fields_internal(vars, |i, j| format!("pub {}: {}", i, j))
}

fn generate_fields_as_args(vars: &Ast) -> Vec<String> {
    generate_fields_internal(vars, |i, j| format!("{}: &mut {}", i, j))
}

#[cfg(feature = "py")]
fn generate_fields_as_immutable_args(vars: &Ast) -> Vec<String> {
    generate_fields_internal(vars, |i, j| format!("{}: {}", i, j))
}

#[cfg(feature = "py")]
fn generate_fields_as_mutable_statements(prefix: &str, vars: &Ast) -> Vec<String> {
    generate_fields_internal(vars, |i, _| format!("let mut {}_{} = {};", prefix, i, i))
}

#[cfg(feature = "py")]
fn generate_fields_as_mutable_refs(prefix: &str, vars: &Ast) -> Vec<String> {
    generate_fields_internal(vars, |i, _| format!("&mut {}_{}", prefix, i))
}

#[cfg(feature = "py")]
fn generate_fields_as_fn_new_args(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    let value = match i {
                        Ast::VariableAssignment { value, .. } => match value {
                            NumOrBool::Number(value) => Ast::Number(*value).generate(),
                            NumOrBool::Bool(value) => format!("{}", value),
                        },
                        _ => unreachable!(),
                    };

                    format!("{}={}", var_name, value)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "py")]
fn generate_py_receptors_as_fn_new_args(kinetics: &str, receptor_vars: &Ast) -> Vec<String> {
    match receptor_vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::Name(name) => name.clone(),
                        _ => unreachable!(),
                    };

                    format!("{}=Py{} {{ receptor: {}::default() }}", var_name, kinetics, kinetics)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "py")]
fn generate_py_receptors_as_args(kinetics: &str, receptor_vars: &Ast) -> Vec<String> {
    match receptor_vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::Name(name) => name.clone(),
                        _ => unreachable!(),
                    };

                    format!("{}: {}", var_name, kinetics)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "py")]
fn generate_py_receptors_as_args_in_receptor(receptor_vars: &Ast) -> Vec<String> {
    match receptor_vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::Name(name) => name.clone(),
                        _ => unreachable!(),
                    };

                    format!("{}: {}.receptor.clone()", var_name, var_name)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "py")]
fn generate_py_basic_gating_vars_as_fn_new_args(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::GatingVariables(variables) => {
            variables
                .iter()
                .map(|i|
                    format!("{}=PyBasicGatingVariable {{ gating_variable: BasicGatingVariable::default() }}", i)
                )
                .collect::<Vec<String>>()
        },
        ast => unreachable!("Unexpected AST in gating variable generation: {:#?}", ast)
    }
}

#[cfg(feature = "py")]
fn generate_py_basic_gating_vars_as_args_in_ion_channel(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::GatingVariables(variables) => {
            variables
                .iter()
                .map(|i|
                    format!("{}: {}.gating_variable.clone()", i, i)
                )
                .collect::<Vec<String>>()
        },
        ast => unreachable!("Unexpected AST in gating variable generation: {:#?}", ast)
    }
}

#[cfg(feature = "py")]
fn generate_fields_as_py_basic_gating_vars_args(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::GatingVariables(variables) => {
            variables
                .iter()
                .map(|i| {
                    format!("{}: PyBasicGatingVariable", i)
                })
                .collect::<Vec<String>>()
        },
        ast => unreachable!("Unexpected AST in gating variable generation: {:#?}", ast)
    }
}

#[cfg(feature = "py")]
fn generate_py_ion_channels_as_fn_new_args(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let (var_name, type_name) = match i {
                        Ast::StructAssignment { name, type_name } => (name, type_name),
                        _ => unreachable!(),
                    };

                    format!("{}=Py{} {{ ion_channel: {}::default() }}", var_name, type_name, type_name)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "py")]
fn generate_py_ion_channels_as_args_in_neuron(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::StructAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    format!("{}: {}.ion_channel.clone()", var_name, var_name)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "py")]
fn generate_py_ion_channels_as_immutable_args(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let (var_name, type_name) = match i {
                        Ast::StructAssignment { name, type_name } => (name, type_name),
                        _ => unreachable!(),
                    };

                    format!("{}: Py{}", var_name, type_name)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

fn generate_fields_as_names(vars: &Ast) -> Vec<String> {
    generate_fields_internal(vars, |i, _| i.to_string())
}

fn generate_fields_as_args_to_pass(vars: &Ast) -> Vec<String> {
    generate_fields_internal(vars, |i, _| format!("&mut self.{}", i))
}

fn replace_self_var(original: String, var: &str, replace_with: &str) -> String {
    let original = original.replace(&format!("self.{}.", var), &format!("{}.", replace_with));
    let original = original.replace(&format!("self.{} ", var), &format!("{} ", replace_with));
    let original = original.replace(&format!("self.{})", var), &format!("{})", replace_with));
    let original = original.replace(&format!("self.{},", var), &format!("{},", replace_with));
    original.replace(&format!("self.{};", var), &format!("{};", replace_with))
}

// make sure to check for `continous()``
fn generate_handle_spiking(on_spike: &Option<Ast>, spike_detection: &Ast) -> String {
    let handle_spiking_header = "fn handle_spiking(&mut self) -> bool {";

    let handle_spiking_function = match on_spike {
        Some(value) => {
            let handle_spiking_check = "\tif self.is_spiking {";
            let handle_spiking_function = format!("\t\t{}", value.generate());

            format!("{}\n{}\n\t}}", handle_spiking_check, handle_spiking_function)
        },
        None => String::from(""),
    };

    if spike_detection.generate() != "continuous()" {
        let handle_is_spiking_calc = format!("\tself.is_spiking = {};", spike_detection.generate());

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
    }
}

#[cfg(feature = "gpu")]
fn generate_gpu_kernel_handle_spiking(on_spike: &Option<Ast>, spike_detection: &Ast) -> String {
    if on_spike.is_none() {
        return String::from("")
    }

    if spike_detection.generate() != "continuous()" {
        format!(
            "if ({}) {{\nis_spiking[index] = 1;\n{}\n}} else {{\nis_spiking[index] = 0;\n}}",  
            spike_detection.generate_kernel_gpu(),
            on_spike.as_ref().unwrap().generate_kernel_gpu(),
        )
    } else {
        let handle_is_spiking_calc = [
            "float last_voltage = current_voltage[index]",
            "uint increasing_right_now = last_voltage < current_voltage[index];",
            "uint threshold_crossed = current_voltage[index] > v_th[index];",
            "is_spiking[index] = threshold_crossed && was_increasing[index] && !increasing_right_now;",
            "was_increasing[index] = increasing_right_now;",
        ];

        format!(
            "{}\n{}",
            handle_is_spiking_calc.join("\n"),
            on_spike.as_ref().unwrap().generate_kernel_gpu(),
        )
    }
}

#[cfg(feature = "gpu")]
fn generate_gpu_kernel_on_iteration(on_iteration: &Ast) -> String {
    let on_iteration_assignments = on_iteration.generate_kernel_gpu();

    let changes = match on_iteration {
        Ast::OnIteration(assignments) => {
            let mut assignments_strings = vec![];

            for i in assignments {
                if let Ast::DiffEqAssignment { name, .. } =  i {
                    let change_string = if name == "v" {
                        "current_voltage[index] += dv;".to_string()
                    } else {
                        format!("{}[index] += d{};", name, name)
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

#[cfg(feature = "gpu")] 
fn generate_kernel_args(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    let type_name = match i {
                        Ast::VariableAssignment { value, .. } => match value {
                            NumOrBool::Number(_) => "float",
                            NumOrBool::Bool(_) => "uint",
                        },
                        _ => unreachable!(),
                    };

                    format!("__global {} *{}", type_name, var_name)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature="gpu")]
fn generate_gating_vars_kernel_args(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::GatingVariables(variables) => {
            variables.iter()
                .map(|i|
                    format!(
                        "String::from(\"{}_alpha\"), 
                        String::from(\"{}_beta\"), 
                        String::from(\"{}_state\")",
                        i,
                        i,
                        i,
                    )
                )
                .collect()
        },
        _ => unreachable!(),
    }
}

#[cfg(feature="gpu")]
fn generate_gating_vars_replacements(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::GatingVariables(variables) => {
            variables.iter()
                .map(|i|
                    format!("
                        update_function = update_function.replace(
                            \"{}$alpha\",
                            &format!(\"{{}}{}_alpha[index]\", prefix)
                        );
                        update_function = update_function.replace(
                            \"{}$beta\",
                            &format!(\"{{}}{}_beta[index]\", prefix)
                        );
                        update_function = update_function.replace(
                            \"{}$state\",
                            &format!(\"{{}}{}_state[index]\", prefix)
                        );",
                        i,
                        i,
                        i,
                        i,
                        i,
                        i,
                    )
                )
                .collect()
        },
        _ => unreachable!()
    }
}

#[cfg(feature="gpu")]
fn generate_gating_vars_update_replacements(vars: &Ast) -> String {
    match vars {
        Ast::GatingVariables(variables) => {
            let replacement_functions: Vec<_> = variables.iter()
                .enumerate()
                .map(|(n, i)|
                    format!(
                        "let replacement_function{} = |args_str: &str| -> String {{
                            format!(
                                \"gating_vars_update(index, $, {{}}{}_alpha, {{}}{}_beta, {{}}{}_state)\", 
                                prefix,
                                prefix,
                                prefix,
                            ).replace(\"$\", args_str)
                        }};",
                        n,
                        i,
                        i,
                        i,
                    )
                )
                .collect();

            let to_replace: Vec<_> = variables.iter()
                .enumerate()
                .map(|(n, i)| 
                    format!("(\"{}$update(\", Box::new(replacement_function{}))", i, n)
                )
                .collect();

            format!("
                {}

                let to_replace: Vec<(_, Box<dyn Fn(&str) -> String>)> = vec![
                    {}
                ];

                for (name_to_replace, replacement_function) in to_replace.iter() {{
                        let mut result = String::new();
                        let mut last_end = 0;

                    while let Some(func_start) = update_function[last_end..].find(name_to_replace) {{
                        let args_start = last_end + func_start + name_to_replace.len() - 1;
                        result.push_str(&update_function[last_end..last_end + func_start]);

                        let remaining_text = &update_function[args_start..];
                        
                        let mut cursor = 1;
                        let mut depth = 1;
                        let mut args_end = 1;
                        
                        while cursor < remaining_text.len() {{
                            match remaining_text.chars().nth(cursor).unwrap() {{
                                '(' => depth += 1,
                                ')' => {{
                                    depth -= 1;
                                    if depth == 0 {{
                                        args_end = cursor;
                                        break;
                                    }}
                                }},
                                _ => {{}}
                            }}
                            cursor += 1;
                        }}

                        let args_str = &remaining_text[1..args_end];
                       
                        result.push_str(&replacement_function(&args_str));
                        
                        last_end = args_start + args_end + 1;
                    }}

                    result.push_str(&update_function[last_end..]);

                    update_function = result;
                }}",
                replacement_functions.join("\n"),
                to_replace.join(",\n"),
            )
        },
        _ => unreachable!()
    }
}

#[cfg(feature="gpu")] 
fn generate_vars_as_arg_strings(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    format!("String::from(\"{}\")", var_name)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature="gpu")] 
fn generate_vars_as_create_buffers(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    let type_name = match i {
                        Ast::VariableAssignment { value, .. } => match value {
                            NumOrBool::Number(_) => "float",
                            NumOrBool::Bool(_) => "uint",
                        },
                        _ => unreachable!(),
                    };

                    format!(
                        "create_{}_buffer!({}_buffer, context, queue, cell_grid, {});", 
                        type_name,
                        var_name,
                        var_name,
                    )
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature="gpu")] 
fn generate_vars_as_insert_buffers(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    let type_name = match i {
                        Ast::VariableAssignment { value, .. } => match value {
                            NumOrBool::Number(_) => "Float",
                            NumOrBool::Bool(_) => "UInt",
                        },
                        _ => unreachable!(),
                    };

                    format!(
                        "buffers.insert(String::from(\"{}\"), BufferGPU::{}({}_buffer));", 
                        var_name,
                        type_name,
                        var_name,
                    )
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_vars_as_field_vecs(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    let (type_name, default_var) = match i {
                        Ast::VariableAssignment { value, .. } => match value {
                            NumOrBool::Number(_) => ("f32", "0.0"),
                            NumOrBool::Bool(_) => ("u32", "0"),
                        },
                        _ => unreachable!(),
                    };

                    format!(
                        "let mut {}: Vec<{}> = vec![{}; rows * cols];", 
                        var_name,
                        type_name,
                        default_var,
                    )
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_gating_vars_as_field_vecs(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::GatingVariables(variables) => {
            variables
                .iter()
                .map(|i| {
                    format!(
                        "let mut {}_alpha: Vec<f32> = vec![0.; rows * cols];
                        let mut {}_beta: Vec<f32> = vec![0.; rows * cols];
                        let mut {}_state: Vec<f32> = vec![0.; rows * cols];", 
                        i,
                        i,
                        i,
                    )
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_vars_as_read_and_set(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    let type_name = match i {
                        Ast::VariableAssignment { value, .. } => match value {
                            NumOrBool::Number(_) => "Float",
                            NumOrBool::Bool(_) => "UInt",
                        },
                        _ => unreachable!(),
                    };

                    format!(
                        "read_and_set_buffer!(buffers, queue, \"{}\", &mut {}, {});", 
                        var_name,
                        var_name,
                        type_name,
                    )
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_vars_as_read_and_set_ion_channel(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    let type_name = match i {
                        Ast::VariableAssignment { value, .. } => match value {
                            NumOrBool::Number(_) => "Float",
                            NumOrBool::Bool(_) => "UInt",
                        },
                        _ => unreachable!(),
                    };

                    format!(
                        "read_and_set_buffer!(buffers, queue, &format!(\"{{}}ion_channel${}\", prefix), &mut {}, {});", 
                        var_name,
                        var_name,
                        type_name,
                    )
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}


#[cfg(feature = "gpu")]
fn generate_gating_vars_as_read_and_set_ion_channel(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::GatingVariables(variables) => {
            variables
                .iter()
                .map(|i|   
                    format!(
                        "read_and_set_buffer!(buffers, queue, &format!(\"{{}}ion_channel${}$alpha\", prefix), &mut {}_alpha, Float);
                        read_and_set_buffer!(buffers, queue, &format!(\"{{}}ion_channel${}$beta\", prefix), &mut {}_beta, Float);
                        read_and_set_buffer!(buffers, queue, &format!(\"{{}}ion_channel${}$state\", prefix), &mut {}_state, Float);", 
                        i,
                        i,
                        i,
                        i,
                        i,
                        i,
                    )
                )
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_vars_as_field_setters(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    match i {
                        Ast::VariableAssignment { value, .. } => match value {
                            NumOrBool::Number(_) => format!("cell.{} = {}[idx];", var_name, var_name),
                            NumOrBool::Bool(_) => format!("cell.{} = {}[idx] == 1;", var_name, var_name),
                        },
                        _ => unreachable!(),
                    }
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_vars_as_ion_channel_field_setters(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    match i {
                        Ast::VariableAssignment { value, .. } => match value {
                            NumOrBool::Number(_) => format!("ion_channel.{} = {}[idx];", var_name, var_name),
                            NumOrBool::Bool(_) => format!("ion_channel.{} = {}[idx] == 1;", var_name, var_name),
                        },
                        _ => unreachable!(),
                    }
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_gating_vars_as_ion_channel_field_setters(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::GatingVariables(variables) => {
            variables
                .iter()
                .map(|i|
                   format!(
                        "ion_channel.{}.alpha = {}_alpha[idx];
                        ion_channel.{}.beta = {}_beta[idx];
                        ion_channel.{}.state = {}_state[idx];", 
                        i, 
                        i,
                        i, 
                        i,
                        i, 
                        i,
                    )
                )
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_ion_channels_to_gpu(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let (var_name, type_name) = match i {
                        Ast::StructAssignment { name, type_name } => (name, type_name),
                        _ => unreachable!(),
                    };

                    format!(
                        "let {}: Vec<Vec<_>> = cell_grid.iter()
                            .map(|row| row.iter().map(|cell| cell.{}.clone()).collect())
                            .collect();
                        let {}_buffers = {}::convert_to_gpu(
                            &{}, context, queue
                        )?;
                        let {}_buffers: HashMap<_, _> = {}_buffers.into_iter()
                            .map(|(k, v)| (format!(\"{}_{{}}\", k), v))
                            .collect();
                        buffers.extend({}_buffers);",
                        var_name,
                        var_name,
                        var_name,
                        type_name,
                        var_name,
                        var_name,
                        var_name,
                        var_name,
                        var_name,
                    )
                })
                .collect()
        },
        _ => unreachable!(),
    }
}

#[cfg(feature = "gpu")]
fn generate_ion_channels_to_cpu(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let (var_name, type_name) = match i {
                        Ast::StructAssignment { name, type_name } => (name, type_name),
                        _ => unreachable!(),
                    };

                    format!("
                        let mut {}s: Vec<Vec<_>> = cell_grid.iter()
                            .map(|row| row.iter().map(|cell| cell.{}.clone()).collect())
                            .collect();

                        {}::convert_to_cpu(
                            \"{}_\", &mut {}s, buffers, rows, cols, queue
                        )?;

                        for (i, row) in cell_grid.iter_mut().enumerate() {{
                            for (j, cell) in row.iter_mut().enumerate() {{
                                cell.{} = {}s[i][j].clone();
                            }}
                        }}",
                        var_name,
                        var_name,
                        type_name,
                        var_name,
                        var_name,
                        var_name,
                        var_name,
                    )
                })
                .collect()
        },
        _ => unreachable!(),
    }
}

// #[cfg(feature = "gpu")]
// fn generate_ion_channel_kernel_args_in_neuron(vars: &Ast) -> Vec<String> {
//     match vars {
//         Ast::StructAssignments(variables) => {
//             variables.iter()
//                 .map(|i| {
//                     let (var_name, type_name) = match i {
//                         Ast::StructAssignment { name, type_name } => (name, type_name),
//                         _ => unreachable!()
//                     };

//                     format!(
//                         "let {}_kernel_args = {}::get_all_attributes_as_vec().iter()
//                             .map(|i| {{
//                                 let type_name = match i.1 {{
//                                     AvailableBufferType::Float => \"float\",
//                                     AvailableBufferType::Float => \"uint\",
//                                     _ => unreachable!(),
//                                 }};
//                                 format!(\"__global {{}} *{}_{{}}\", type_name, i.0.split(\"$\").join(\"_\"))
//                             }})
//                             .collect::<Vec<_>>();",
//                         var_name,
//                         type_name,
//                         var_name,
//                     )
//                 })
//                 .collect()
//         },
//         _ => unreachable!(),
//     }
// }

#[cfg(feature = "gpu")]
fn generate_ion_channel_argument_names_for_neuron_kernel(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let (var_name, type_name) = match i {
                        Ast::StructAssignment { name, type_name } => (name, type_name),
                        _ => unreachable!(),
                    };

                    format!(
                        "argument_names.extend(
                            {}::get_attribute_names_as_vector().iter()
                                .map(|i| format!(\"{}_{{}}\", i.0))
                                .collect::<Vec<_>>()
                        );",
                        type_name,
                        var_name,
                    )
                })
                .collect()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_ion_channel_args_names(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let name = match i {
                        Ast::StructAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    format!("{}_args", name)
                })
                .collect()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_ion_channel_attrs_getter(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let (var_name, type_name) = match i {
                        Ast::StructAssignment { name, type_name } => (name, type_name),
                        _ => unreachable!(),
                    };

                    format!(
                        "let {}_args = {}::get_attribute_names_as_vector().iter().map(|i| i.0.clone()).collect::<Vec<_>>();",
                        var_name,
                        type_name,
                    )
                })
                .collect()
        },
        _ => unreachable!(),
    }
}

#[cfg(feature = "gpu")]
fn generate_ion_channel_prefixes(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::StructAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    format!(
                        "let {}_prefix = generate_unique_prefix(&argument_names, \"{}_ion_channel\");",
                        var_name,
                        var_name,
                    )
                })
                .collect()
        },
        _ => unreachable!(),
    }
}

#[cfg(feature = "gpu")]
fn generate_ion_channel_replacements_in_neuron_kernel(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let (var_name, type_name) = match i {
                        Ast::StructAssignment { name, type_name } => (name, type_name),
                        _ => unreachable!(),
                    };

                    format!(
                        "for (attr, _) in {}::get_attribute_names_as_vector() {{
                            let current_split: Vec<_> = attr.split(\"$\").collect();
                            program_source = program_source.replace(
                                &format!(\"{}${{}}\", &current_split[1..].join(\"$\")),
                                &format!(\"{{}}{{}}[index]\", {}_prefix, &current_split[1..].join(\"$\")),
                            );
                        }}",
                        type_name,
                        var_name,
                        var_name,
                    )
                })
                .collect()
        },
        _ => unreachable!(),
    }
}

#[cfg(feature = "gpu")]
fn generate_ion_channel_replacements_in_neuron_kernel_header(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let (var_name, type_name) = match i {
                        Ast::StructAssignment { name, type_name } => (name, type_name),
                        _ => unreachable!(),
                    };

                    format!(
                        "program_source = program_source.replace(
                            \"@{}_args\", 
                            {}::get_attribute_names_as_vector()
                            .iter().map(|i| {{
                                let type_name = match i.1 {{
                                    AvailableBufferType::Float => \"float\",
                                    AvailableBufferType::UInt => \"uint\",
                                    AvailableBufferType::OptionalUInt => \"int\",
                                }};

                                format!(
                                    \"__global {{}} *{{}}{{}}\", 
                                    type_name, 
                                    {}_prefix, 
                                    (&i.0.split(\"$\").collect::<Vec<_>>()[1..]).join(\"\")
                                )
                            }})
                            .collect::<Vec<_>>()
                            .join(\",\n\")
                            .as_str()
                        );",
                        var_name,
                        type_name,
                        var_name,
                    )
                })
                .collect()
        }, 
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_ion_channel_get_function_calls_to_replace(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let name = match i {
                        Ast::StructAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    format!("@{}_function", name)
                })
                .collect()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_ion_channel_get_function_calls(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let (var_name, type_name) = match i {
                        Ast::StructAssignment { name, type_name } => (name, type_name),
                        _ => unreachable!(),
                    };

                    format!(
                        "program_source = program_source.replace(\"@{}_function\", {}::get_update_function().1.as_str());", 
                        var_name,
                        type_name,
                    )
                })
                .collect()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_ion_channel_replace_call_in_neuron_kernel(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::StructAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let (var_name, type_name) = match i {
                        Ast::StructAssignment { name, type_name } => (name, type_name),
                        _ => unreachable!(),
                    };

                    format!(
                        "let update_current_name = \"{}$update_current(\";
                        let update_current_replacement_function = |args: &Vec<&str>| -> String {{
                            format!(
                                \"update_{}_ion_channel(index, {{}}, {{}})\", 
                                args[0],
                                {}::get_attribute_names_as_vector()
                                    .iter().map(|i| {{
                                        format!(
                                            \"{{}}{{}}\", 
                                            {}_prefix, 
                                            (&i.0.split(\"$\").collect::<Vec<_>>()[1..]).join(\"\")
                                        )
                                    }})
                                    .collect::<Vec<_>>()
                                    .join(\", \")
                                    .as_str()
                            )
                        }};

                        let mut result = String::new();
                        let mut last_end = 0;

                        while let Some(func_start) = program_source[last_end..].find(update_current_name) {{
                            let args_start = last_end + func_start + update_current_name.len() - 1;
                            result.push_str(&program_source[last_end..last_end + func_start]);

                            let remaining_text = &program_source[args_start..];
                            
                            let mut cursor = 1;
                            let mut depth = 1;
                            let mut args_end = 1;
                            
                            while cursor < remaining_text.len() {{
                                match remaining_text.chars().nth(cursor).unwrap() {{
                                    '(' => depth += 1,
                                    ')' => {{
                                        depth -= 1;
                                        if depth == 0 {{
                                            args_end = cursor;
                                            break;
                                        }}
                                    }},
                                    _ => {{}}
                                }}
                                cursor += 1;
                            }}

                            let args_str = &remaining_text[1..args_end];
                            let args = args_str.split(\",\").collect::<Vec<_>>();
                            
                            result.push_str(&update_current_replacement_function(&args));
                            
                            last_end = args_start + args_end + 1;
                        }}

                        result.push_str(&program_source[last_end..]);

                        program_source = result;",
                        var_name,
                        type_name, 
                        type_name,
                        var_name,
                    )
                })
                .collect()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "py")]
fn generate_py_getter_and_setters(field_name: &str, var_name: &str, type_name: &str) -> String {
    format!("
        #[getter]
        fn get_{}(&self) -> {} {{
            self.{}.{}
        }}

        #[setter]
        fn set_{}(&mut self, new_param: {}) {{
            self.{}.{} = new_param;
        }}",
        var_name,
        type_name,
        field_name,
        var_name,
        var_name,
        type_name,
        field_name,
        var_name,
    )
}

#[cfg(feature = "py")]
fn generate_vars_as_getter_setters(field_name: &str, vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        ast => unreachable!("Unreachable AST on individual variable: {:#?}", ast),
                    };

                    match i {
                        Ast::VariableAssignment { value, .. } => {
                            let type_name = match value {
                                NumOrBool::Number(_) => "f32",
                                NumOrBool::Bool(_) => "bool",
                            };

                            generate_py_getter_and_setters(field_name, var_name, type_name)
                        }
                        ast => unreachable!("Unreachable AST on individual variable: {:#?}", ast),
                    }
                })
                .collect()
        },
        ast => unreachable!("Unreachable AST on variable assignments: {:#?}", ast),
    }
}

#[cfg(feature = "py")]
fn generate_receptor_vars_as_getter_setters(type_name: &str, vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables.iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::Name(name) => name.clone(),
                        ast => unreachable!("Unreachable AST on individual variable in receptors parsing: {:#?}", ast),
                    };

                    format!("
                        #[getter]
                        fn get_{}(&self) -> {} {{
                            {} {{ receptor: self.receptor.{} }}
                        }}

                        #[setter]
                        fn set_{}(&mut self, new_param: {}) {{
                            self.receptor.{} = new_param.receptor;
                        }}",
                        var_name,
                        type_name,
                        type_name,
                        var_name,
                        var_name,
                        type_name,
                        var_name,
                    )
                })
                .collect()
        },
        ast => unreachable!("Unreachable AST on receptor variables parsing: {:#?}", ast),
    }
}

impl NeuronDefinition {
    // eventually adapt for documentation to be integrated
    // for now use default ligand gates and neurotransmitter implementation
    // if defaults come with vars assignment then add default trait
    // if neurotransmitter kinetics and receptor kinetics specified then
    // create default_impl() function
    fn to_code(&self) -> (Vec<String>, String) {
        let (neurotransmitter_kinetics, receptor_kinetics) = if let Some(Ast::KineticsDefinition(neuro, receptor)) = &self.kinetics {
            (neuro.clone(), receptor.clone())
        } else {
            (String::from("ApproximateNeurotransmitter"), String::from("ApproximateReceptor"))
        };

        let receptors_name = match &self.receptors {
            Some(val) => val.generate(),
            None => String::from("DefaultReceptors"),
        };

        let neurotransmitter_kind = format!("{}NeurotransmitterType", receptors_name);

        let mut imports = vec![
            String::from("use spiking_neural_networks::neuron::iterate_and_spike_traits::IterateAndSpikeBase;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::Receptors;"),
        ];

        if self.kinetics.is_none() {
            imports.push(
                String::from(
                    "use spiking_neural_networks::neuron::iterate_and_spike::ApproximateNeurotransmitter;"
                )
            );
            imports.push(
                String::from(
                    "use spiking_neural_networks::neuron::iterate_and_spike::ApproximateReceptor;"
                )
            );
        }

        let neuron_necessary_imports = [
            "CurrentVoltage", "GapConductance", "LastFiringTime", "IsSpiking",
            "Timestep", "IterateAndSpike", 
            "Neurotransmitters", "NeurotransmitterKinetics", "ReceptorKinetics",
            "NeurotransmitterConcentrations"
        ];

        for i in neuron_necessary_imports {
            imports.push(format!("use spiking_neural_networks::neuron::iterate_and_spike::{};", i));
        }

        if self.receptors.is_none() {
            imports.push(
                String::from("use spiking_neural_networks::neuron::iterate_and_spike::DefaultReceptors;")
            );
            imports.push(
                format!(
                    "use spiking_neural_networks::neuron::iterate_and_spike::{};",
                    neurotransmitter_kind,
                )
            );
        }

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
        let receptors_field = format!("pub receptors: {}<R>", receptors_name);

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
        defaults.push(format!("receptors: {}::<R>::default()", receptors_name));

        let default_fields = defaults.join(",\n\t");
        
        let default_function = format!(
            "fn default() -> Self {{ {} {{\n\t{}\n}}", 
            self.type_name.generate(),
            default_fields,
        );
        let default_function = add_indents(&default_function, "\t");

        let impl_default = format!(
            "\nimpl<T: NeurotransmitterKinetics, R: ReceptorKinetics> Default for {}<T, R> {{\n\t{}\n}}\n}}\n",
            self.type_name.generate(),
            default_function,
        );

        let impl_default_impl = format!(
            "\nimpl {}<{}, {}> {{ pub fn default_impl() -> Self {{ Self::default() }} }}",
            self.type_name.generate(),
            neurotransmitter_kinetics,
            receptor_kinetics,
        );

        let handle_spiking = generate_handle_spiking(&self.on_spike, &self.spike_detection);

        let get_concentrations_header = "fn get_neurotransmitter_concentrations(&self) -> NeurotransmitterConcentrations<Self::N> {";
        let get_concentrations_body = "self.synaptic_neurotransmitters.get_concentrations()";
        let get_concentrations_function = format!("{}\n\t{}\n}}", get_concentrations_header, get_concentrations_body);

        let handle_neurotransmitter_conc = "self.synaptic_neurotransmitters.apply_t_changes(&NeurotransmittersIntermediate::from_neuron(self));";
        let handle_spiking_call = "self.handle_spiking()";
        let iteration_body = format!(
            "\n\t{}\n\t{}", 
            generate_on_iteration(&self.on_iteration), 
            handle_spiking_call,
        );
        let iteration_function = format!("{}{}\n}}", ITERATION_HEADER, iteration_body);

        let iteration_with_neurotransmitter_header = generate_iteration_with_neurotransmitter_header();

        let receptors_update = "self.receptors.update_receptor_kinetics(t, self.dt)";
        let receptors_set_current = "self.receptors.set_receptor_currents(self.current_voltage, self.dt)";
        let receptors_get_current = "self.receptors.get_receptor_currents(self.dt, self.c_m)";

        let iteration_with_neurotransmitter_function = match &self.on_electrochemical_iteration {
            Some(val) => {
                let iteration = generate_on_iteration(val);
                let iteration = replace_self_var(iteration, "input_current", "input_current");
                let iteration = replace_self_var(iteration, "t", "t");

                let iteration = replace_self_var(
                    iteration, 
                    "synaptic_neurotransmitters.apply_t_changes()", 
                    handle_neurotransmitter_conc,
                );

                let iteration_body = format!(
                    "{}\n{}",
                    iteration,
                    handle_spiking_call,
                );

                if iteration.contains("self.receptors.set_receptor_currents") || 
                    iteration.contains("self.receptors.get_receptor_currents") {
                    imports.push(
                        String::from(
                            "use spiking_neural_networks::neuron::iterate_and_spike::IonotropicReception;"
                        )
                    );
                }
                if iteration.contains("self.synaptic_neurotransmitters.apply_t_changes") {
                    imports.push(
                        String::from(
                            "use spiking_neural_networks::neuron::intermediate_delegate::NeurotransmittersIntermediate;"
                        )
                    );
                }

                format!(
                    "{}\n{}\n}}",
                    iteration_with_neurotransmitter_header,
                    iteration_body,
                )
            },
            None => {
                let update_with_receptor_current = format!(
                    "self.current_voltage -= {};",
                    receptors_get_current,
                );
        
                let iteration_with_neurotransmitter_body = format!(
                    "\t{};\n\t{};\n\t{}\n\t{}\n\t{}\n\t{}",
                    receptors_update,
                    receptors_set_current,
                    generate_on_iteration(&self.on_iteration),
                    update_with_receptor_current,
                    handle_neurotransmitter_conc,
                    handle_spiking_call,
                );

                imports.push(
                    String::from(
                        "use spiking_neural_networks::neuron::iterate_and_spike::IonotropicReception;"
                    )
                );
                imports.push(
                    String::from(
                        "use spiking_neural_networks::neuron::intermediate_delegate::NeurotransmittersIntermediate;"
                    )
                );

                format!(
                    "{}\n{}\n}}", 
                    iteration_with_neurotransmitter_header,
                    iteration_with_neurotransmitter_body,
                )
            }
        };

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
            imports,
            format!(
                "{}\n{}\n{}\n}}\n\n{}\n\n{}\n{}\n{}", 
                macros, 
                header, 
                fields, 
                impl_functions, 
                impl_iterate_and_spike,
                impl_default,
                impl_default_impl,
            )
        )
    }

    #[cfg(feature = "gpu")]
    fn to_gpu_code(&self) -> (Vec<String>, String) {
        let impl_header = format!("impl<T: NeurotransmitterKineticsGPU, R: ReceptorKineticsGPU> IterateAndSpikeGPU for {}<T, R> {{", self.type_name.generate());

        let iterate_and_spike_electrical_kernel_header = "fn iterate_and_spike_electrical_kernel(context: &Context) -> Result<KernelFunction, GPUError> {";
        let kernel_name = "let kernel_name = String::from(\"iterate_and_spike\");"; 
        let mandatory_variables = [
            ("current_voltage", "float"), 
            ("dt", "float"), 
            ("is_spiking", "uint"), 
            ("gap_conductance", "float"), 
            ("c_m", "float"),
        ];
        
        let iterate_and_spike_kernel_footer = "
            let iterate_and_spike_program = match Program::create_and_build_from_source(context, &program_source, \"\") {
                Ok(value) => value,
                Err(_) => return Err(GPUError::ProgramCompileFailure),
            };
            let kernel = match Kernel::create(&iterate_and_spike_program, &kernel_name) {
                Ok(value) => value,
                Err(_) => return Err(GPUError::KernelCompileFailure),
            };

            Ok(
                KernelFunction { 
                    kernel, 
                    program_source, 
                    kernel_name, 
                    argument_names, 
                }
            )\n}";

        // iterate over ion channel vars
        // use them to replace vars
        // replace update func with actual function, only add func if it is found in kernel
        
        let ion_channel_args = generate_ion_channel_attrs_getter(self.ion_channels.as_ref().unwrap_or(&Ast::StructAssignments(vec![])));
        let ion_channel_prefixes = generate_ion_channel_prefixes(self.ion_channels.as_ref().unwrap_or(&Ast::StructAssignments(vec![])));
        let ion_channel_var_replacements = generate_ion_channel_replacements_in_neuron_kernel(self.ion_channels.as_ref().unwrap_or(&Ast::StructAssignments(vec![])));
        let ion_channel_args_names = generate_ion_channel_args_names(self.ion_channels.as_ref().unwrap_or(&Ast::StructAssignments(vec![])));

        let ion_channel_get_function_calls_to_replace = generate_ion_channel_get_function_calls_to_replace(self.ion_channels.as_ref().unwrap_or(&Ast::StructAssignments(vec![]))).join("\n");
        let ion_channel_get_function_calls_replacements = generate_ion_channel_get_function_calls(self.ion_channels.as_ref().unwrap_or(&Ast::StructAssignments(vec![]))).join(", ");

        let ion_channel_header_replacements = generate_ion_channel_replacements_in_neuron_kernel_header(
            self.ion_channels.as_ref().unwrap_or(&Ast::StructAssignments(vec![]))
        ).join("\n");

        let ion_channel_function_call_replacements_in_kernel = generate_ion_channel_replace_call_in_neuron_kernel(
            self.ion_channels.as_ref().unwrap_or(&Ast::StructAssignments(vec![]))
        ).join("\n");

        let ion_channel_argument_names_extensions = generate_ion_channel_argument_names_for_neuron_kernel(
            self.ion_channels.as_ref().unwrap_or(&Ast::StructAssignments(vec![]))
        ).join("\n");

        let ion_channel_prefixes = ion_channel_prefixes.join("\n");
        let ion_channel_kernel_args = ion_channel_args.join("\n");
        let ion_channel_kernel_args_replacements = ion_channel_var_replacements.join("\n");

        let iterate_and_spike_electrical_function = if ion_channel_prefixes.is_empty() {    
            let argument_names = format!(
                "let argument_names = vec![String::from(\"inputs\"), String::from(\"index_to_position\"), {}, {}];",
                mandatory_variables.iter().map(|i| format!("String::from(\"{}\")", i.0)).collect::<Vec<String>>().join(","),
                generate_vars_as_arg_strings(&self.vars).join(", "),
            );
            
            let kernel_header = format!(
                "__kernel void iterate_and_spike(
                    __global const float *inputs,
                    __global const uint *index_to_position,
                    {},
                    {}
                ) {{
                    int gid = get_global_id(0);
                    int index = index_to_position[gid];",
                mandatory_variables.iter().map(|i| format!("__global {} *{}", i.1, i.0)).collect::<Vec<String>>().join(",\n"),
                generate_kernel_args(&self.vars).join(",\n"),
            );

            let kernel_body = format!(
                "{}\n{}",
                generate_gpu_kernel_on_iteration(&self.on_iteration), 
                generate_gpu_kernel_handle_spiking(&self.on_spike, &self.spike_detection),
            );

            let kernel = format!("let program_source = \"{}\n{}\n}}\".to_string();", kernel_header, kernel_body);

            format!(
                "{}\n{}\n{}\n{}\n{}", 
                iterate_and_spike_electrical_kernel_header, 
                kernel_name,
                argument_names,
                kernel,
                iterate_and_spike_kernel_footer,
            )
        } else {
            let argument_names = format!(
                "let mut argument_names = vec![String::from(\"inputs\"), String::from(\"index_to_position\"), {}, {}];
                {}",
                mandatory_variables.iter().map(|i| format!("String::from(\"{}\")", i.0)).collect::<Vec<String>>().join(","),
                generate_vars_as_arg_strings(&self.vars).join(", "),
                ion_channel_argument_names_extensions,
            );

            let kernel_header = format!(
                "__kernel void iterate_and_spike(
                    __global const float *inputs,
                    __global const uint *index_to_position,
                    {},
                    {},
                    {}
                ) {{
                    int gid = get_global_id(0);
                    int index = index_to_position[gid];",
                mandatory_variables.iter().map(|i| format!("__global {} *{}", i.1, i.0)).collect::<Vec<String>>().join(",\n"),
                generate_kernel_args(&self.vars).join(",\n"),
                ion_channel_args_names.iter().map(|i| format!("@{}", i)).collect::<Vec<String>>().join(",\n"),
            );

            let kernel_body = format!(
                "{}\n{}",
                generate_gpu_kernel_on_iteration(&self.on_iteration), 
                generate_gpu_kernel_handle_spiking(&self.on_spike, &self.spike_detection),
            );
    
            let kernel = format!(
                "let mut program_source = \"{}\n{}\n{}\n}}\".to_string();", 
                ion_channel_get_function_calls_to_replace,
                kernel_header, 
                kernel_body,
            );

            format!(
                "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}", 
                iterate_and_spike_electrical_kernel_header, 
                kernel_name,
                argument_names,
                ion_channel_prefixes,
                ion_channel_kernel_args,
                kernel,
                ion_channel_kernel_args_replacements,
                ion_channel_header_replacements,
                ion_channel_function_call_replacements_in_kernel,
                ion_channel_get_function_calls_replacements,
                iterate_and_spike_kernel_footer,
            )
        };

        // generate electrochemical kernel
        // if no chemical iteration, add neurotransmitter update, receptor kinetics, receptor update to end
        // if chemical iteration, replace associated functions with gpu functions
        // start with synaptic neurotransmitter update
        // for receptor kinetics iterate through receptor attrs list and use that to determine what
        // receptor kinetics functions to call
        // then replace receptor update function

        let receptors_name = match &self.receptors {
            Some(val) => val.generate(),
            None => String::from("DefaultReceptors"),
        };

        let iterate_and_spike_electrochemical_header = "fn iterate_and_spike_electrochemical_kernel(context: &Context) -> Result<KernelFunction, GPUError> {";

        let argument_names = format!(
            "let mut argument_names = vec![
                String::from(\"number_of_types\"), String::from(\"inputs\"), String::from(\"t\"),
                String::from(\"index_to_position\"), String::from(\"neurotransmitters$flags\"), String::from(\"receptors$flags\"),
                {}, {}
            ];
            argument_names.extend(
                T::get_attribute_names_as_vector().iter().map(|(i, _)| i.clone()).collect::<Vec<_>>()
            );",
            mandatory_variables.iter().map(|i| format!("String::from(\"{}\")", i.0)).collect::<Vec<String>>().join(","),
            generate_vars_as_arg_strings(&self.vars).join(", "),
        );

        let neurotransmitter_vars_generation = "
            let neuro_prefix = generate_unique_prefix(&argument_names, \"neurotransmitters\");
            let neurotransmitter_args = T::get_attribute_names_as_vector()
                .iter()
                .map(|i| (
                    i.1, 
                    format!(
                        \"{}{}\", neuro_prefix,
                        i.0.split(\"$\").collect::<Vec<&str>>()[1],
                    )
                ))
                .collect::<Vec<(AvailableBufferType, String)>>();
            let neurotransmitter_arg_names = neurotransmitter_args.iter()
                .map(|i| i.1.clone())
                .collect::<Vec<String>>();
            let neurotransmitter_arg_and_type = neurotransmitter_args.iter()
                .map(|(i, j)| 
                    format!(
                    \"__global {} *{}\", 
                    match i { 
                        AvailableBufferType::Float => \"float\",
                        AvailableBufferType::UInt => \"uint\",
                        _ => unreachable!(),
                    }, 
                    j
                ))
                .collect::<Vec<String>>();
        ";

        // generate set currents function from receptors get updates
        // iterate over all types to only update them when receptor flags are enabled
        // check which have current variables, if they have current variables
        // use them in calculation of get receptor currents
    
        let receptors_vars_generation = format!("
            let receptor_prefix = generate_unique_prefix(
                &argument_names,
                \"receptors\"
            );
            let receptor_kinetics_prefix = generate_unique_prefix(
                &argument_names,
                \"kinetics_receptors\"
            );

            let mut receptor_arg_and_type = vec![];
            let mut receptor_kinetics_args: HashMap<(String, String), Vec<String>> = HashMap::new();
            for (i, j) in {}::<R>::get_all_attributes().iter() {{
                let current_split = i.split(\"$\").collect::<Vec<&str>>();
                if current_split.len() == 2 {{
                    let current_type = match &j {{
                        AvailableBufferType::Float => \"float\",
                        AvailableBufferType::UInt => \"uint\",
                        _ => unreachable!(),
                    }};
                    receptor_arg_and_type.push(
                        format!(\"__global {{}} *{{}}{{}}\", current_type, receptor_prefix, current_split[1])
                    );
                }} else {{
                    let current_arg = format!(
                        \"{{}}{{}}_{{}}_{{}}\", 
                        receptor_kinetics_prefix, 
                        current_split[1], 
                        current_split[2], 
                        current_split[4]
                    );
                    receptor_kinetics_args.entry((current_split[1].to_string(), current_split[2].to_string()))
                        .or_default()
                        .push(current_arg.clone());

                    let current_type = match &j {{
                        AvailableBufferType::Float => \"float\",
                        AvailableBufferType::UInt => \"uint\",
                        _ => unreachable!(),
                    }};
                    receptor_arg_and_type.push(
                        format!(\"__global {{}} *{{}}\", current_type, current_arg)
                    );
                }}

                argument_names.push(i.clone());
            }}

            let mut conversion: HashMap<String, usize> = HashMap::new();
            for i in <{}::<R> as Receptors>::N::get_all_types().iter() {{
                conversion.insert(i.to_string(), i.type_to_numeric());
            }}

            let mut update_receptor_kinetics = vec![];
            for ((neuro, name), _) in receptor_kinetics_args.iter() {{
                let update = format!(
                    \"
                    if (receptors_flags[index * number_of_types + {{}}] == 1) {{{{
                        {{}}{{}}_{{}}_r[index] = get_r(t[index * number_of_types + {{}}], dt[index], {{}});
                    }}}}\", 
                    conversion.get(neuro).unwrap(), 
                    receptor_kinetics_prefix, 
                    neuro,
                    name,
                    conversion.get(neuro).unwrap(), 
                    R::get_update_function().0.iter()
                        .filter(|i| i.starts_with(\"receptors\"))
                        .map(|i| format!(
                            \"{{}}{{}}_{{}}_{{}}[index]\", 
                            receptor_kinetics_prefix, 
                            neuro,
                            name,
                            i.split(\"$\").collect::<Vec<_>>().last().unwrap()
                        ))
                        .collect::<Vec<_>>().join(\", \")
                );
                update_receptor_kinetics.push(update);
            }}

            let mut receptor_updates = vec![];
            for (update, update_args) in <{}::<R> as ReceptorsGPU>::get_updates().iter() {{
                let current_prefix = \"__kernel void update_\";
                let rest = &update[current_prefix.len()..];
                let end = rest.find('(').unwrap();
                let neuro = &rest[..end];

                let mut current_args = vec![];
                for arg in update_args.iter() {{
                    let current_split = arg.0.split(\"$\").collect::<Vec<_>>();
                    if current_split.len() == 2 {{
                        current_args.push(
                            format!(
                                \"{{}}{{}}\", 
                                receptor_prefix, 
                                current_split.last().unwrap()
                            )
                        );
                    }} else if current_split.len() > 2 {{
                        current_args.push(
                            format!(
                                \"{{}}{{}}_{{}}_{{}}\",
                                receptor_kinetics_prefix,
                                current_split[1], 
                                current_split[2],
                                current_split[4],
                            )
                        );
                    }}
                }}
                
                let current_update = format!(
                    \"
                    if (receptors_flags[index * number_of_types + {{}}] == 1) {{{{
                        update_{{}}(index, current_voltage, dt, {{}});
                    }}}}\", 
                    conversion.get(neuro).unwrap(),
                    neuro, 
                    current_args.join(\", \"),
                );
                receptor_updates.push(current_update);
            }}

            let mut current_attrs = vec![];
            let mut current_neuros = vec![];
            for (i, _) in {}::<R>::get_all_attributes().iter() {{
                let current_split = i.split(\"$\").collect::<Vec<&str>>();
                if current_split.len() == 2 {{
                    for neuro in conversion.keys() {{
                        if format!(\"{{}}_current\", neuro) == *current_split.last().unwrap() {{
                            current_attrs.push(format!(\"{{}}{{}}_current\", receptor_prefix, neuro));
                            current_neuros.push(neuro.clone());
                        }}
                    }}
                }}
            }}
            let get_currents = current_attrs.iter()
                .zip(current_neuros.iter())
                .map(|(i, j)| format!(
                    \"(((float) receptors_flags[index * number_of_types + {{}}]) * {{}}[index])\", 
                    conversion.get(j).unwrap(),
                    i,
                ))
                .collect::<Vec<_>>()
                .join(\" + \");
            ",
            receptors_name,
            receptors_name,
            receptors_name,
            receptors_name,
        );

        let neurotransmitters_update_code = String::from("
            neurotransmitters_update(
                index, 
                number_of_types,
                neuro_flags,
                current_voltage,
                is_spiking,
                dt,
                {}
            );"
        );
        
        let update_receptors_replace = "
            let kinetics_name = \"receptors$update_receptor_kinetics(\";
            let set_receptor_currents_name = \"receptors$set_receptor_currents(\";
            let get_receptor_currents_name = \"receptors$get_receptor_currents(\";

            let kinetics_replacement = |args: &Vec<&str>| -> String { 
                update_receptor_kinetics.join(\"\n\").replace(\"dt[index]\", args[1])
            };
            let set_receptor_currents_replacement = |args: &Vec<&str>| -> String {
                receptor_updates.join(\"\n\").replace(\"current_voltage[index]\", args[0])
                    .replace(\"dt[index]\", args[1])
            };
            let get_receptor_currents_replacement = |args: &Vec<&str>| -> String {
                format!(\"({} / {}) * {}\", args[0], args[1], get_currents)
            };

            let to_replace: Vec<(_, Box<dyn Fn(&Vec<&str>) -> String>)> = vec![
                (kinetics_name, Box::new(kinetics_replacement)), 
                (set_receptor_currents_name, Box::new(set_receptor_currents_replacement)),
                (get_receptor_currents_name, Box::new(get_receptor_currents_replacement)),
            ];

            for (name_to_replace, replacement_function) in to_replace.iter() {
                let mut result = String::new();
                let mut last_end = 0;

                while let Some(func_start) = program_source[last_end..].find(name_to_replace) {
                    let args_start = last_end + func_start + name_to_replace.len() - 1;
                    result.push_str(&program_source[last_end..last_end + func_start]);

                    let remaining_text = &program_source[args_start..];
                    
                    let mut cursor = 1;
                    let mut depth = 1;
                    let mut args_end = 1;
                    
                    while cursor < remaining_text.len() {
                        match remaining_text.chars().nth(cursor).unwrap() {
                            '(' => depth += 1,
                            ')' => {
                                depth -= 1;
                                if depth == 0 {
                                    args_end = cursor;
                                    break;
                                }
                            },
                            _ => {}
                        }
                        cursor += 1;
                    }

                    let args_str = &remaining_text[1..args_end];
                    let args = args_str.split(\",\").collect::<Vec<_>>();
                    
                    result.push_str(&replacement_function(&args));
                    
                    last_end = args_start + args_end + 1;
                }

                result.push_str(&program_source[last_end..]);

                program_source = result;
            }";

        let iterate_and_spike_electrochemical_function = match (&self.on_electrochemical_iteration, &self.ion_channels.is_some()) {
            (Some(body), true) => {
                let kernel_header = format!(
                    "__kernel void iterate_and_spike(
                        uint number_of_types,
                        __global const float *inputs,
                        __global const float *t,
                        __global const uint *index_to_position,
                        __global const uint *neuro_flags,
                        __global const uint *receptors_flags,
                        {},
                        {},
                        {},
                        {{}},
                        {{}}
                    ) {{{{
                        int gid = get_global_id(0);
                        int index = index_to_position[gid];",
                    mandatory_variables.iter().map(|i| format!("__global {} *{}", i.1, i.0)).collect::<Vec<String>>().join(",\n"),
                    generate_kernel_args(&self.vars).join(",\n"),
                    ion_channel_args_names.iter().map(|i| format!("@{}", i)).collect::<Vec<String>>().join(",\n"),
                );

                let kernel_body = format!(
                    "{}\n{}",
                    generate_gpu_kernel_on_iteration(body).replace("{", "{{").replace("}", "}}"), 
                    generate_gpu_kernel_handle_spiking(&self.on_spike, &self.spike_detection).replace("{", "{{").replace("}", "}}"),
                );

                let neurotransmitters_replace = format!(
                    "let neurotransmitter_replace = format!(\"{}\", neurotransmitter_arg_names.join(\",\n\"));", 
                    neurotransmitters_update_code,
                );
               
                let kernel = format!("
                    {}
                    let mut program_source = format!(
                        \"{{}}\n{{}}\n{{}}\n{{}}\n{}\n{}\n{}\n}}}}\", 
                        R::get_update_function().1,
                        T::get_update_function().1, 
                        <{}<R> as ReceptorsGPU>::get_updates().iter().map(|i| i.0.clone()).collect::<Vec<_>>().join(\"\n\"),
                        Neurotransmitters::<<{}<R> as Receptors>::N, T>::get_neurotransmitter_update_kernel_code(),
                        neurotransmitter_arg_and_type.join(\",\n\"),
                        receptor_arg_and_type.join(\",\n\"),
                    ).replace(\"synaptic_neurotransmitters$apply_t_changes();\", &neurotransmitter_replace);
                    
                    {}", 
                    neurotransmitters_replace,
                    ion_channel_get_function_calls_to_replace,
                    kernel_header, 
                    kernel_body,
                    receptors_name,
                    receptors_name,
                    update_receptors_replace,
                );

                format!(
                    "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}", // \nprintln!(\"{{}}\", program_source);
                    iterate_and_spike_electrochemical_header, 
                    kernel_name,
                    argument_names,
                    ion_channel_argument_names_extensions,
                    neurotransmitter_vars_generation,
                    receptors_vars_generation,
                    ion_channel_prefixes,
                    ion_channel_kernel_args,
                    kernel,
                    ion_channel_kernel_args_replacements,
                    ion_channel_header_replacements,
                    ion_channel_function_call_replacements_in_kernel,
                    ion_channel_get_function_calls_replacements,
                    iterate_and_spike_kernel_footer,
                )
            },
            (Some(body), false) => {
                let kernel_header = format!(
                    "__kernel void iterate_and_spike(
                        uint number_of_types,
                        __global const float *inputs,
                        __global const float *t,
                        __global const uint *index_to_position,
                        __global const uint *neuro_flags,
                        __global const uint *receptors_flags,
                        {},
                        {},
                        {{}},
                        {{}}
                    ) {{{{
                        int gid = get_global_id(0);
                        int index = index_to_position[gid];",
                    mandatory_variables.iter().map(|i| format!("__global {} *{}", i.1, i.0)).collect::<Vec<String>>().join(",\n"),
                    generate_kernel_args(&self.vars).join(",\n"),
                );

                let kernel_body = format!(
                    "{}\n{}",
                    generate_gpu_kernel_on_iteration(body).replace("{", "{{").replace("}", "}}"), 
                    generate_gpu_kernel_handle_spiking(&self.on_spike, &self.spike_detection).replace("{", "{{").replace("}", "}}"),
                );

                let neurotransmitters_replace = format!(
                    "let neurotransmitter_replace = format!(\"{}\", neurotransmitter_arg_names.join(\",\n\"));", 
                    neurotransmitters_update_code,
                );

                let kernel = format!(
                    "
                    {}
                    let mut program_source = format!(
                        \"{{}}\n{{}}\n{{}}\n{{}}\n{}\n{}\n}}}}\", 
                        R::get_update_function().1,
                        T::get_update_function().1, 
                        <{}<R> as ReceptorsGPU>::get_updates().iter().map(|i| i.0.clone()).collect::<Vec<_>>().join(\"\n\"),
                        Neurotransmitters::<<{}<R> as Receptors>::N, T>::get_neurotransmitter_update_kernel_code(),
                        neurotransmitter_arg_and_type.join(\",\n\"),
                        receptor_arg_and_type.join(\",\n\"),
                    ).replace(\"synaptic_neurotransmitters$apply_t_changes();\", &neurotransmitter_replace);
                    
                    {}", 
                    neurotransmitters_replace,
                    kernel_header, 
                    kernel_body,
                    receptors_name,
                    receptors_name,
                    update_receptors_replace,
                );

                format!(
                    "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}", // \nprintln!(\"{{}}\", program_source);
                    iterate_and_spike_electrochemical_header, 
                    kernel_name,
                    argument_names,
                    neurotransmitter_vars_generation,
                    receptors_vars_generation,
                    ion_channel_prefixes,
                    ion_channel_kernel_args,
                    kernel,
                    ion_channel_kernel_args_replacements,
                    ion_channel_header_replacements,
                    ion_channel_function_call_replacements_in_kernel,
                    ion_channel_get_function_calls_replacements,
                    iterate_and_spike_kernel_footer,
                )
            },
            (None, true) => {
                let kernel_header = format!(
                    "__kernel void iterate_and_spike(
                        uint number_of_types,
                        __global const float *inputs,
                        __global const float *t,
                        __global const uint *index_to_position,
                        __global const uint *neuro_flags,
                        __global const uint *receptors_flags,
                        {},
                        {},
                        {},
                        {{}},
                        {{}}
                    ) {{{{
                        int gid = get_global_id(0);
                        int index = index_to_position[gid];",
                    mandatory_variables.iter().map(|i| format!("__global {} *{}", i.1, i.0)).collect::<Vec<String>>().join(",\n"),
                    generate_kernel_args(&self.vars).join(",\n"),
                    ion_channel_args_names.iter().map(|i| format!("@{}", i)).collect::<Vec<String>>().join(",\n"),
                );

                let kernel_body = format!(
                    "{{}}\n{{}}\n{}\n{{}}\n{}\n{}",
                    generate_gpu_kernel_on_iteration(&self.on_iteration).replace("{", "{{").replace("}", "}}"), 
                    neurotransmitters_update_code,
                    generate_gpu_kernel_handle_spiking(&self.on_spike, &self.spike_detection).replace("{", "{{").replace("}", "}}"),
                );

                let kernel = format!(
                    "let program_source = format!(
                        \"{{}}\n{{}}\n{{}}\n{{}}\n{}\n{}\n{}\n}}}}\", 
                        R::get_update_function().1,
                        T::get_update_function().1, 
                        <{}<R> as ReceptorsGPU>::get_updates().iter().map(|i| i.0.clone()).collect::<Vec<_>>().join(\"\n\"),
                        Neurotransmitters::<<{}<R> as Receptors>::N, T>::get_neurotransmitter_update_kernel_code(),
                        neurotransmitter_arg_and_type.join(\",\n\"),
                        receptor_arg_and_type.join(\",\n\"),
                        update_receptor_kinetics.join(\"\n\"),
                        receptor_updates.join(\"\n\"),
                        format!(\"current_voltage[index] -= (dt[index] / c_m[index]) * ({{}});\", get_currents),
                        neurotransmitter_arg_names.join(\",\n\"),
                    );", 
                    ion_channel_get_function_calls_to_replace,
                    kernel_header, 
                    kernel_body,
                    receptors_name,
                    receptors_name,
                );

                format!(
                    "{}\n{}\n{}\n{}\n{}\n{}\n{}", // \nprintln!(\"{{}}\", program_source);
                    iterate_and_spike_electrochemical_header, 
                    kernel_name,
                    argument_names,
                    neurotransmitter_vars_generation,
                    receptors_vars_generation,
                    kernel,
                    iterate_and_spike_kernel_footer,
                )
            },
            (None, false) => {
                let kernel_header = format!(
                    "__kernel void iterate_and_spike(
                        uint number_of_types,
                        __global const float *inputs,
                        __global const float *t,
                        __global const uint *index_to_position,
                        __global const uint *neuro_flags,
                        __global const uint *receptors_flags,
                        {},
                        {},
                        {{}},
                        {{}}
                    ) {{{{
                        int gid = get_global_id(0);
                        int index = index_to_position[gid];",
                    mandatory_variables.iter().map(|i| format!("__global {} *{}", i.1, i.0)).collect::<Vec<String>>().join(",\n"),
                    generate_kernel_args(&self.vars).join(",\n"),
                );

                let kernel_body = format!(
                    "{{}}\n{{}}\n{}\n{{}}\n{}\n{}",
                    generate_gpu_kernel_on_iteration(&self.on_iteration).replace("{", "{{").replace("}", "}}"), 
                    neurotransmitters_update_code,
                    generate_gpu_kernel_handle_spiking(&self.on_spike, &self.spike_detection).replace("{", "{{").replace("}", "}}"),
                );

                let kernel = format!(
                    "let program_source = format!(
                        \"{{}}\n{{}}\n{{}}\n{{}}\n{}\n{}\n}}}}\", 
                        R::get_update_function().1,
                        T::get_update_function().1, 
                        <{}<R> as ReceptorsGPU>::get_updates().iter().map(|i| i.0.clone()).collect::<Vec<_>>().join(\"\n\"),
                        Neurotransmitters::<<{}<R> as Receptors>::N, T>::get_neurotransmitter_update_kernel_code(),
                        neurotransmitter_arg_and_type.join(\",\n\"),
                        receptor_arg_and_type.join(\",\n\"),
                        update_receptor_kinetics.join(\"\n\"),
                        receptor_updates.join(\"\n\"),
                        format!(\"current_voltage[index] -= (dt[index] / c_m[index]) * ({{}});\", get_currents),
                        neurotransmitter_arg_names.join(\",\n\"),
                    );", 
                    kernel_header, 
                    kernel_body,
                    receptors_name,
                    receptors_name,
                );

                format!(
                    "{}\n{}\n{}\n{}\n{}\n{}\n{}", // \nprintln!(\"{{}}\", program_source);
                    iterate_and_spike_electrochemical_header, 
                    kernel_name,
                    argument_names,
                    neurotransmitter_vars_generation,
                    receptors_vars_generation,
                    kernel,
                    iterate_and_spike_kernel_footer,
                )
            },
        };

        let ion_channels_to_gpu = generate_ion_channels_to_gpu(
            self.ion_channels.as_ref().unwrap_or(&Ast::StructAssignments(vec![]))
        );

        // let iterate_and_spike_electrochemical_function = "fn iterate_and_spike_electrochemical_kernel(context: &Context) -> Result<KernelFunction, GPUError> { todo!() }";
        let convert_to_gpu = format!("
            fn convert_to_gpu(
                cell_grid: &[Vec<Self>], 
                context: &Context,
                queue: &CommandQueue,
            ) -> Result<HashMap<String, BufferGPU>, GPUError> {{
                if cell_grid.is_empty() || cell_grid.iter().all(|i| i.is_empty()) {{
                    return Ok(HashMap::new());
                }}

                let mut buffers = HashMap::new();

                {}

                {}

                create_optional_uint_buffer!(last_firing_time_buffer, context, queue, cell_grid, last_firing_time, last);

                {}

                {}

                buffers.insert(String::from(\"last_firing_time\"), BufferGPU::OptionalUInt(last_firing_time_buffer));

                {}

                Ok(buffers)
            }}
            ",
            mandatory_variables.iter()
                .map(|(i, j)| 
                    format!("create_{}_buffer!({}_buffer, context, queue, cell_grid, {});", j, i, i)
                )
                .collect::<Vec<String>>()
                .join("\n"),
            generate_vars_as_create_buffers(&self.vars).join("\n"),
            mandatory_variables.iter()
                .map(|(i, j)| 
                    format!(
                        "buffers.insert(String::from(\"{}\"), BufferGPU::{}({}_buffer));", 
                        i, 
                        if *j == "float" { "Float" } else { "UInt" }, 
                        i
                    )
                )
                .collect::<Vec<String>>()
                .join("\n"),
            generate_vars_as_insert_buffers(&self.vars).join("\n"),
            ion_channels_to_gpu.join("\n"),
        );

        let ion_channels_to_cpu = generate_ion_channels_to_cpu(
            self.ion_channels.as_ref().unwrap_or(&Ast::StructAssignments(vec![]))
        );

        let convert_to_cpu = format!("
            fn convert_to_cpu(
                cell_grid: &mut Vec<Vec<Self>>,
                buffers: &HashMap<String, BufferGPU>,
                rows: usize,
                cols: usize,
                queue: &CommandQueue,
            ) -> Result<(), GPUError> {{ 
                if rows == 0 || cols == 0 {{
                    cell_grid.clear();

                    return Ok(());
                }}

                {}

                {}

                let mut last_firing_time: Vec<i32> = vec![0; rows * cols];

                {}

                {}

                read_and_set_buffer!(buffers, queue, \"last_firing_time\", &mut last_firing_time, OptionalUInt);

                for i in 0..rows {{
                    for j in 0..cols {{
                        let idx = i * cols + j;
                        let cell = &mut cell_grid[i][j];
                        
                        {}

                        {}

                        cell.last_firing_time = if last_firing_time[idx] == -1 {{
                            None
                        }} else {{
                            Some(last_firing_time[idx] as usize)
                        }};
                    }}
                }}

                {}

                Ok(())
            }}
            ",
            mandatory_variables.iter()
                .map(|(i, j)| 
                    format!(
                        "let mut {}: Vec<{}> = vec![{}; rows * cols];", 
                        i, 
                        if *j == "float" { "f32" } else { "u32" },
                        if *j == "float" { "0.0" } else { "0" },
                    )
                )
                .collect::<Vec<String>>()
                .join("\n"),
            generate_vars_as_field_vecs(&self.vars).join("\n"),
            mandatory_variables.iter()
                .map(|(i, j)| 
                    format!(
                        "read_and_set_buffer!(buffers, queue, \"{}\", &mut {}, {});", 
                        i, 
                        i,
                        if *j == "float" { "Float" } else { "UInt" },
                    )
                )
                .collect::<Vec<String>>()
                .join("\n"),
            generate_vars_as_read_and_set(&self.vars).join("\n"),
            mandatory_variables.iter()
                .map(|(i, j)| 
                    if *j == "float" {
                        format!("cell.{} = {}[idx];", i, i)
                    } else {
                        format!("cell.{} = {}[idx] == 1;", i, i)
                    }
                )
                .collect::<Vec<String>>()
                .join("\n"),
            generate_vars_as_field_setters(&self.vars).join("\n"),
            ion_channels_to_cpu.join("\n"),
        );
        
        let convert_electrochemical_to_gpu = format!("
            fn convert_electrochemical_to_gpu(
                cell_grid: &[Vec<Self>], 
                context: &Context,
                queue: &CommandQueue,
            ) -> Result<HashMap<String, BufferGPU>, GPUError> {{
                if cell_grid.is_empty() {{
                    return Ok(HashMap::new());
                }}

                let mut buffers = Self::convert_to_gpu(cell_grid, context, queue)?;

                let neurotransmitters: Vec<Vec<_>> = cell_grid.iter()
                    .map(|row| row.iter().map(|cell| cell.synaptic_neurotransmitters.clone()).collect())
                    .collect();
                let receptors: Vec<Vec<_>> = cell_grid.iter()
                    .map(|row| row.iter().map(|cell| cell.receptors.clone()).collect())
                    .collect();

                let neurotransmitter_buffers = Neurotransmitters::<<{}::<R> as Receptors>::N, T>::convert_to_gpu(
                    &neurotransmitters, context, queue
                )?;
                let receptors_buffers = {}::<R>::convert_to_gpu(
                    &receptors, context, queue
                )?;

                buffers.extend(neurotransmitter_buffers);
                buffers.extend(receptors_buffers);

                Ok(buffers)
            }}",
            receptors_name,
            receptors_name,
        );
        
        let convert_electrochemical_to_cpu = format!("
            fn convert_electrochemical_to_cpu(
                cell_grid: &mut Vec<Vec<Self>>,
                buffers: &HashMap<String, BufferGPU>,
                rows: usize,
                cols: usize,
                queue: &CommandQueue,
            ) -> Result<(), GPUError> {{ 
                if rows == 0 || cols == 0 {{
                    cell_grid.clear();

                    return Ok(());
                }}

                let mut neurotransmitters: Vec<Vec<_>> = cell_grid.iter()
                    .map(|row| row.iter().map(|cell| cell.synaptic_neurotransmitters.clone()).collect())
                    .collect();
                let mut receptors: Vec<Vec<_>> = cell_grid.iter()
                    .map(|row| row.iter().map(|cell| cell.receptors.clone()).collect())
                    .collect();

                Self::convert_to_cpu(cell_grid, buffers, rows, cols, queue)?;
                
                Neurotransmitters::<<{}::<R> as Receptors>::N, T>::convert_to_cpu(
                    &mut neurotransmitters, buffers, queue, rows, cols
                )?;
                {}::<R>::convert_to_cpu(
                    &mut receptors, buffers, queue, rows, cols
                )?;

                for (i, row) in cell_grid.iter_mut().enumerate() {{
                    for (j, cell) in row.iter_mut().enumerate() {{
                        cell.synaptic_neurotransmitters = neurotransmitters[i][j].clone();
                        cell.receptors = receptors[i][j].clone();
                    }}
                }}

                Ok(())
            }}",
            receptors_name,
            receptors_name,
        );

        let imports = vec![
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::IterateAndSpikeGPU;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::NeurotransmitterKineticsGPU;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::ReceptorKineticsGPU;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::ReceptorsGPU;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::NeurotransmitterTypeGPU;"),
            String::from("use spiking_neural_networks::error::GPUError;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::KernelFunction;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::BufferGPU;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::AvailableBufferType;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::create_float_buffer;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::create_uint_buffer;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::create_optional_uint_buffer;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::write_buffer;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::read_and_set_buffer;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::flatten_and_retrieve_field;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::generate_unique_prefix;"),
            String::from("use std::collections::HashMap;"),
            String::from("use std::ptr;"),
            String::from("use opencl3::memory::Buffer;"),
            String::from("use opencl3::memory::CL_MEM_READ_WRITE;"),
            String::from("use opencl3::types::CL_BLOCKING;"),
            String::from("use opencl3::types::CL_NON_BLOCKING;"),
            String::from("use opencl3::types::cl_float;"),
            String::from("use opencl3::types::cl_uint;"),
            String::from("use opencl3::types::cl_int;"),
            String::from("use opencl3::context::Context;"),
            String::from("use opencl3::program::Program;"),
            String::from("use opencl3::kernel::Kernel;"),
            String::from("use opencl3::command_queue::CommandQueue;"),
        ];

        (
            imports,
            format!(
                "
                {}
                {}
                {}
                {}
                {}
                {}
                {}
                }}
                ",
                impl_header,
                iterate_and_spike_electrical_function,
                iterate_and_spike_electrochemical_function,
                convert_to_gpu,
                convert_to_cpu,
                convert_electrochemical_to_gpu,
                convert_electrochemical_to_cpu,
            ),
        )
    }

    #[cfg(feature = "py")]
    fn to_pyo3_code(&self) -> (Vec<String>, String) {
        let (neurotransmitter_kinetics, receptor_kinetics) = if let Some(Ast::KineticsDefinition(neuro, receptor)) = &self.kinetics {
            (neuro.clone(), receptor.clone())
        } else {
            (String::from("ApproximateNeurotransmitter"), String::from("ApproximateReceptor"))
        };

        let struct_def = format!("
            #[pyclass]
            #[pyo3(name = \"{}\")]
            #[derive(Clone)]
            pub struct Py{} {{
                model: {}<{}, {}>,
            }}",
            self.type_name.generate(),
            self.type_name.generate(),
            self.type_name.generate(),
            neurotransmitter_kinetics,
            receptor_kinetics,
        );

        // iterate to generate getter and setters
        let mandatory_vars = [
            ("dt", "f32"), 
            ("current_voltage", "f32"), 
            ("c_m", "f32"), 
            ("gap_conductance", "f32"), 
            ("is_spiking", "bool"),
        ];

        let defaults = vec![
            String::from("dt=0.1"), String::from("current_voltage=0."), String::from("c_m=1."), 
            String::from("gap_conductance=10."), String::from("is_spiking=false"),
        ];

        let receptors_name = self.receptors.as_ref()
            .unwrap_or(&Ast::TypeDefinition(String::from("DefaultReceptors")))
            .generate();

        let constructor = format!(
            "#[new]
            #[pyo3(signature = ({}, {}, {} synaptic_neurotransmitters=None, receptors=Py{} {{ receptors: {}::default() }}))]
            fn new({}, {}, {} synaptic_neurotransmitters: Option<&PyDict>, receptors: Py{}) -> PyResult<Self> {{ 
                let model = {} {{
                    {},
                    {},
                    {}
                    ..{}::default()
                }};

                let mut neuron = Py{} {{ model }};
                if let Some(dict) = synaptic_neurotransmitters {{
                    neuron.set_synaptic_neurotransmitters(dict)?
                }};
                neuron.set_receptors(receptors);

                Ok(neuron)
            }}",
            defaults.join(", "),
            generate_fields_as_fn_new_args(&self.vars).join(", "),
            if let Some(ion_channels) = &self.ion_channels {
                format!("{},", generate_py_ion_channels_as_fn_new_args(&ion_channels).join(", "))
            } else {
                String::from("")
            },
            receptors_name,
            receptors_name,
            mandatory_vars.iter().map(|(i, j)| format!("{}: {}", i, j)).collect::<Vec<_>>().join(", "),
            generate_fields_as_immutable_args(&self.vars).join(", "),
            if let Some(ion_channels) = &self.ion_channels {
                format!("{},", generate_py_ion_channels_as_immutable_args(&ion_channels).join(", "))
            } else {
                String::from("")
            },
            receptors_name,
            self.type_name.generate(),
            mandatory_vars.iter().map(|(i, _)| i.to_string()).collect::<Vec<_>>().join(", "),
            generate_fields_as_names(&self.vars).join(", "),
            if let Some(ion_channels) = &self.ion_channels {
                format!("{},", generate_py_ion_channels_as_args_in_neuron(&ion_channels).join(", "))
            } else {
                String::from("")
            },
            self.type_name.generate(),
            self.type_name.generate(),
        );

        let repr = r#"fn __repr__(&self) -> PyResult<String> { Ok(format!("{:#?}", self.model)) }"#;

        let mandatory_getter_and_setters: Vec<String> = mandatory_vars.iter()
            .map(|(i, j)| generate_py_getter_and_setters("model", i, j))
            .collect();

        let mut basic_getter_setters = generate_vars_as_getter_setters("model", &self.vars);  
        basic_getter_setters.extend(mandatory_getter_and_setters);   

        let get_and_set_last_firing_time = "
            #[getter(last_firing_time)]
            fn get_last_firing_time(&self) -> Option<usize> {
                self.model.get_last_firing_time()
            }

            #[setter(last_firing_time)]
            fn set_last_firing_time(&mut self, timestep: Option<usize>) {
                self.model.set_last_firing_time(timestep);
            }";

        let iterate_and_spike_function = "fn iterate_and_spike(&mut self, i: f32) -> bool {
            self.model.iterate_and_spike(i)
        }";

        // generate new method from default var assignments

        // synaptic neurotransmitters as pydict of neurotransmitter enum to kinetics structs
        // need neurotransmitter to py type conversion in associated types
        // or maybe just use the conversion function, maybe conversion function needs to be
        // a 2-way conversion function

        let neurotransmitters_getter_and_setter = format!(
            "fn get_synaptic_neurotransmitters<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {{
                let dict = PyDict::new(py);
                for (key, value) in self.model.synaptic_neurotransmitters.iter() {{
                    let key_py = Py::new(py, key.convert_type_to_py())?;
                    let val_py = Py::new(py, Py{} {{
                        neurotransmitter: value.clone(),
                    }})?;
                    dict.set_item(key_py, val_py)?;
                }}

                Ok(dict)
            }}

            fn set_synaptic_neurotransmitters(&mut self, neurotransmitters: &PyDict) -> PyResult<()> {{
                let current_copy = self.model.synaptic_neurotransmitters.clone();
                let keys: Vec<_> = self.model.synaptic_neurotransmitters.keys().cloned().collect();
                for key in keys.iter() {{
                    self.model.synaptic_neurotransmitters.remove(key).unwrap();
                }}

                for (key, value) in neurotransmitters.iter() {{
                    let current_type = <{}<{}> as Receptors>::N::convert_from_py(key);
                    if current_type.is_none() {{
                        self.model.synaptic_neurotransmitters = current_copy;
                        return Err(PyTypeError::new_err(\"Incorrect neurotransmitter type\"));
                    }}
                    let current_neurotransmitter = value.extract::<Py{}>();
                    if current_neurotransmitter.is_err() {{
                        self.model.synaptic_neurotransmitters = current_copy;
                        return Err(PyTypeError::new_err(\"Incorrect neurotransmitter kinetics type\"));
                    }}
                    self.model.synaptic_neurotransmitters.insert(
                        current_type.unwrap(), 
                        current_neurotransmitter.unwrap().neurotransmitter.clone(),
                    );
                }}

                Ok(())
            }}",
            neurotransmitter_kinetics,
            self.receptors.as_ref()
                .unwrap_or(&Ast::TypeDefinition(String::from("DefaultReceptors")))
                .generate(),
            receptor_kinetics,
            neurotransmitter_kinetics,
        );

        let receptors_getter_and_setter = format!(
            "fn get_receptors(&self) -> Py{} {{
                Py{} {{ receptors: self.model.receptors.clone() }}
            }}
            
            fn set_receptors(&mut self, receptors: Py{}) {{
                self.model.receptors = receptors.receptors.clone();
            }}",
            receptors_name,
            receptors_name,
            receptors_name,
        );

        let electrochemical_iterate_and_spike_function = format!(
            "fn iterate_with_neurotransmitter_and_spike(&mut self, input_current: f32, t: &PyDict) -> PyResult<()> {{
                let mut conc = HashMap::new();
                for (key, value) in t.iter() {{
                    let current_type = {}NeurotransmitterType::convert_from_py(key);
                    if current_type.is_none() {{
                        return Err(PyTypeError::new_err(\"Incorrect neurotransmitter type\"));
                    }}
                    let current_t = value.extract::<f32>()?;
                    conc.insert(
                        current_type.unwrap(), 
                        current_t
                    );
                }}
                
                self.model.iterate_with_neurotransmitter_and_spike(input_current, &conc);

                Ok(())
            }}",
            receptors_name,
        );

        let ion_channels_getters_setters = match &self.ion_channels {
            Some(Ast::StructAssignments(variables)) => {
                variables.iter()
                    .map(|i| {
                        let (var_name, type_name) = match i {
                            Ast::StructAssignment { name, type_name } => (name, type_name),
                            _ => unreachable!(),
                        };

                        format!(
                            "fn get_{}(&self) -> Py{} {{
                                Py{} {{ ion_channel: self.model.{}.clone() }}
                            }}

                            fn set_{}(&mut self, new_param: Py{}) {{
                                self.model.{} = new_param.ion_channel.clone();
                            }}",
                            var_name,
                            type_name,
                            type_name,
                            var_name,
                            var_name,
                            type_name,
                            var_name,
                        )
                    })
                    .collect::<Vec<String>>()
            },
            None => vec![],
            _ => unreachable!()
        };

        let impl_pymethods = format!(
            "
            #[pymethods]
            impl Py{} {{
                {}
                {}
                {}
                {}
                {}
                {}
                {}
                {}
                {}
            }}
            ",
            self.type_name.generate(),
            constructor,
            repr,
            basic_getter_setters.join("\n"),
            get_and_set_last_firing_time,
            neurotransmitters_getter_and_setter,
            receptors_getter_and_setter,
            ion_channels_getters_setters.join("\n"),
            iterate_and_spike_function,
            electrochemical_iterate_and_spike_function,
        );

        let imports = vec![
            String::from("use pyo3::prelude::*;"),
            String::from("use pyo3::types::PyDict;"),
            String::from("use pyo3::exceptions::PyTypeError;"),
        ];

        (
            imports,
            format!(
                "
                {}
                {}
                ",
                struct_def,
                impl_pymethods,
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

fn parse_kinetics_definition(pair: Pair<'_, Rule>) -> (String, Ast) {
    let mut nested_rule = pair.into_inner();
    let neuro = String::from(nested_rule.next().unwrap().as_str());
    let receptor = String::from(nested_rule.next().unwrap().as_str());

    (
        String::from("kinetics_def"),
        Ast::KineticsDefinition(neuro, receptor)
    )
}

fn parse_receptor_params_def(pair: Pair<'_, Rule>) -> (String, Ast) {
    (
        String::from("receptors_param_def"), 
        Ast::TypeDefinition(
            String::from(pair.into_inner().next().unwrap().as_str())
        )
    )
}

fn parse_iteration_internals(pair: Pair<'_, Rule>) -> Ast {
    let inner_rules = pair.into_inner();

    Ast::OnIteration(
        inner_rules
        .map(|i| parse_declaration(i))
        .collect::<Vec<Ast>>()
    )
}

fn parse_on_iteration(pair: Pair<'_, Rule>) -> (String, Ast) {
    (
        String::from("on_iteration"),
        parse_iteration_internals(pair)
    )
}

fn parse_on_electrochemical_iteration(pair: Pair<'_, Rule>) -> (String, Ast) {
    (
        String::from("on_electrochemical_iteration"),
        parse_iteration_internals(pair)
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

fn parse_vars_with_default(pair: Pair<'_, Rule>) -> (String, Ast) {
    let inner_rules = pair.into_inner();

    let assignments: Vec<Ast> = inner_rules 
        .map(|i| {
            let mut nested_rule = i.into_inner();

            Ast::VariableAssignment { 
                name: String::from(nested_rule.next().unwrap().as_str()), 
                value: {
                    let parsed = nested_rule.next()
                        .unwrap();

                    match parsed.as_str().parse::<f32>() {
                        Ok(val) => NumOrBool::Number(val),
                        Err(_) => NumOrBool::Bool(parsed.as_str().parse::<bool>().unwrap())
                    }
                }
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
            Rule::kinetics_def => {
                parse_kinetics_definition(pair)
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
            Rule::vars_with_default_def => {
                parse_vars_with_default(pair)
            },
            Rule::ion_channels_def => {
                parse_ion_channels(pair)
            },
            Rule::receptors_param_def => {
                parse_receptor_params_def(pair)
            },
            Rule::on_electrochemical_iteration_def => {
                parse_on_electrochemical_iteration(pair)
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
    
    let kinetics = definitions.remove("kinetics_def");
    let on_spike = definitions.remove("on_spike");
    let ion_channels = definitions.remove("ion_channels");
    let receptors = definitions.remove("receptors_param_def");
    let on_electrochemical_iteration = definitions.remove("on_electrochemical_iteration");

    Ok(
        NeuronDefinition {
            type_name,
            kinetics,
            vars,
            spike_detection,
            on_iteration,
            on_spike,
            ion_channels,
            on_electrochemical_iteration,
            receptors,
        }
    )
}

fn generate_defaults(vars: &Ast) -> Vec<String> {
    let defaults = match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    if let Ast::VariableAssignment { name, value } = i {
                        match value {
                            NumOrBool::Number(x) => {
                                if x % 1. == 0. {
                                    format!("{}: {}.", name, x)
                                } else {
                                    format!("{}: {}", name, x)
                                }
                            },
                            NumOrBool::Bool(x) => format!("{}: {}", name, x)
                        }
                    } else {
                        unreachable!("Expected variable assignment")
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
                    } else if let Ast::StructCall { args, .. } = i {
                        if args.is_none() {
                            continue;
                        }

                        for arg in args.as_ref().unwrap() {
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
            "#[derive(Debug, Clone, Copy, PartialEq)]\npub struct {} {{", 
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

            let update_current_body = generate_on_iteration(&self.on_iteration);

            let update_current_body = replace_self_var(update_current_body, "current_voltage", "voltage");
            let update_current_body = replace_self_var(update_current_body, "dt", "dt");

            format!(
                "{}\n{}\n}}", 
                update_current_header, 
                update_current_body,
            )
        } else {
            let update_current_header = "fn update_current(&mut self, voltage: f32) {";
            let update_current_body = add_indents(&self.on_iteration.generate(), "\t");
            let update_current_body = replace_self_var(update_current_body, "current_voltage", "voltage");

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

    #[cfg(feature = "gpu")]
    fn to_gpu_code(&self) -> (Vec<String>, String) {
        // when used in neuron, ion channels should be prefixed with their name
        // in order to avoid naming conflicts
        // after prefixing, they can be added to the whole buffer hashmap

        let gating_vars_attrs = generate_gpu_gating_vars_attributes_vec(
            self.gating_vars.as_ref().unwrap_or(&Ast::GatingVariables(vec![]))
        );

        let get_all_attrs_as_vec = format!(
            "fn get_attribute_names_as_vector() -> Vec<(String, AvailableBufferType)> {{
                vec![(String::from(\"ion_channel$current\"), AvailableBufferType::Float), {} {}]
            }}",
            generate_gpu_ion_channel_attributes_vec(&self.vars).join(", "),
            if !gating_vars_attrs.is_empty() {
                format!(",{}", gating_vars_attrs.join(", "))
            } else {
                String::from("")
            },
        );

        let get_all_attrs = "fn get_all_attributes() -> HashSet<(String, AvailableBufferType)> {
            Self::get_attribute_names_as_vector().into_iter().collect()
        }";

        let get_gating_vars = generate_gpu_gating_vars_attribute_matching(
            self.gating_vars.as_ref().unwrap_or(&Ast::GatingVariables(vec![]))
        );

        let get_attribute_header = "fn get_attribute(&self, value: &str) -> Option<BufferType> {";
        let get_attribute_body = format!(
            "match value {{ \"ion_channel$current\" => Some(BufferType::Float(self.current)),\n{},{}\n_ => None }}", 
            generate_gpu_ion_channel_attribute_matching(&self.vars).join(",\n"),
            if !get_gating_vars.is_empty() {
                format!("{},\n", get_gating_vars.join(",\n"))
            } else {
                String::from("")
            },
        );

        let get_attribute = format!("{}\n{}\n}}", get_attribute_header, get_attribute_body);

        let set_gating_vars = generate_gpu_gating_vars_attribute_setting(
            self.gating_vars.as_ref().unwrap_or(&Ast::GatingVariables(vec![]))
        );

        let set_attribute_header = "fn set_attribute(&mut self, attribute: &str, value: BufferType) -> Result<(), std::io::Error> {";
        let set_current_attribute = "\"ion_channel$current\" => self.current = match value {
                BufferType::Float(nested_val) => nested_val,
                _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, \"Invalid type\")),
            }";
        let set_attribute_body = format!(
            "match attribute {{ {},\n{},{}\n_ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, \"Invalid attribute\")) }};\nOk(())",
            set_current_attribute,
            generate_gpu_ion_channel_attribute_setting(&self.vars).join(",\n"),
            if !set_gating_vars.is_empty() {
                format!("{},\n", set_gating_vars.join(",\n"))
            } else {
                String::from("")
            },
        );

        let set_attribute = format!("{}\n{}\n}}", set_attribute_header, set_attribute_body);

        let gating_vars_conversion_to_cpu = if !gating_vars_attrs.is_empty() {
            format!(
                "{}

                {}

                for i in 0..rows {{
                    for j in 0..cols {{
                        let idx = i * cols + j;
                        let ion_channel = &mut grid[i][j];
                        {}
                    }}
                }}",
                generate_gating_vars_as_field_vecs(self.gating_vars.as_ref().unwrap()).join("\n"),
                generate_gating_vars_as_read_and_set_ion_channel(self.gating_vars.as_ref().unwrap()).join("\n"),
                generate_gating_vars_as_ion_channel_field_setters(self.gating_vars.as_ref().unwrap()).join("\n"),
            )
        } else {
            String::from("")
        };

        let convert_to_cpu = format!("fn convert_to_cpu(
                prefix: &str,
                grid: &mut Vec<Vec<Self>>,
                buffers: &HashMap<String, BufferGPU>,
                rows: usize,
                cols: usize,
                queue: &CommandQueue,
            ) -> Result<(), GPUError> {{ 
                if rows == 0 || cols == 0 {{
                    grid.clear();

                    return Ok(());
                }}

                let mut current: Vec<f32> = vec![0.0; rows * cols];

                {}

                read_and_set_buffer!(buffers, queue, &format!(\"{{}}ion_channel$current\", prefix), &mut current, Float);

                {}

                for i in 0..rows {{
                    for j in 0..cols {{
                        let idx = i * cols + j;
                        let ion_channel = &mut grid[i][j];
                        
                        ion_channel.current = current[idx];

                        {}
                    }}
                }}

                {}

                Ok(())
            }}",
            generate_vars_as_field_vecs(&self.vars).join("\n"),
            generate_vars_as_read_and_set_ion_channel(&self.vars).join("\n"),
            generate_vars_as_ion_channel_field_setters(&self.vars).join("\n"),
            gating_vars_conversion_to_cpu,
        );

        let convert_to_gpu = "fn convert_to_gpu(
            grid: &[Vec<Self>], context: &Context, queue: &CommandQueue
        ) -> Result<HashMap<String, BufferGPU>, GPUError> {
            if grid.is_empty() || grid.iter().all(|i| i.is_empty()) {
                return Ok(HashMap::new());
            }

            let mut buffers = HashMap::new();

            let size: usize = grid.iter().map(|row| row.len()).sum();
            
            for (attr, current_type) in Self::get_all_attributes() {
                match current_type {
                    AvailableBufferType::Float => {
                        let mut current_attrs: Vec<f32> = vec![];
                        for row in grid.iter() {
                            for i in row.iter() {
                                match i.get_attribute(&attr) {
                                    Some(BufferType::Float(val)) => current_attrs.push(val),
                                    Some(_) => unreachable!(),
                                    None => current_attrs.push(0.),
                                };
                            }
                        }

                        write_buffer!(current_buffer, context, queue, size, &current_attrs, Float, last);

                        buffers.insert(attr.clone(), BufferGPU::Float(current_buffer));
                    },
                    AvailableBufferType::UInt => {
                        let mut current_attrs: Vec<u32> = vec![];
                        for row in grid.iter() {
                            for i in row.iter() {
                                match i.get_attribute(&attr) {
                                    Some(BufferType::UInt(val)) => current_attrs.push(val),
                                    Some(_) => unreachable!(),
                                    None => current_attrs.push(0),
                                };
                            }
                        }

                        write_buffer!(current_buffer, context, queue, size, &current_attrs, UInt, last);

                        buffers.insert(attr.clone(), BufferGPU::UInt(current_buffer));
                    },
                    _ => unreachable!(),
                }
            }
            
            Ok(buffers)
        }";

        let gating_vars_attrs_no_types = generate_gpu_gating_vars_attributes_vec_no_types(
            self.gating_vars.as_ref().unwrap_or(&Ast::GatingVariables(vec![]))
        );

        let update_function = if gating_vars_attrs.is_empty() {
            format!(
                "fn get_update_function() -> (Vec<String>, String) {{
                    (
                        vec![{}, {}],
                        String::from(\"__kernel void update_{}_ion_channel(
                            uint index,
                            {},
                            {}
                        ) {{
                            {}
                        }}\"){}
                    )
                }}",
                if self.get_use_timestep() {
                    "String::from(\"current_voltage\"), String::from(\"ion_channel$dt\"), String::from(\"ion_channel$current\")"
                } else {
                    "String::from(\"current_voltage\"), String::from(\"ion_channel$current\")"
                },
                generate_gpu_ion_channel_attributes_vec_no_types(&self.vars).join(", "),
                self.type_name.generate(),
                if self.get_use_timestep() {
                    "float current_voltage,\nfloat dt,\n__global float *current"
                } else {
                    "float current_voltage,\n__global float *current"
                },
                generate_kernel_args(&self.vars).join(",\n"),
                generate_gpu_kernel_on_iteration(&self.on_iteration),
                if self.get_use_timestep() {
                    ".replace(\"current_voltage[index]\", \"current_voltage\").replace(\"dt[index]\", \"dt\")"
                } else {
                    ".replace(\"current_voltage[index]\", \"current_voltage\")"
                }
            )
        } else {
            let kernel_on_iteration = generate_gpu_kernel_on_iteration(&self.on_iteration);

            let update_gating_vars_func = "__kernel void gating_vars_update(
                uint index,
                float dt,
                __global float *alpha,
                __global float *beta,
                __global float *state
            ) {{
                state[index] = dt * (alpha[index] * (1.0f - state[index]) - (beta[index] * state[index]));
            }}";

            format!(
                "fn get_update_function() -> (Vec<String>, String) {{
                    let kernel_args = vec![{}];
                    let gating_vars_args = vec![{}];
                    let prefix = generate_unique_prefix(&kernel_args, \"gating_vars\");
                    let gating_vars_args: Vec<_> = gating_vars_args.iter()
                        .map(|i| format!(\"__global float *{{}}{{}}\", prefix, i))
                        .collect();
                    
                    let mut update_function = format!(\"{}
                    
                        __kernel void update_{}_ion_channel(
                            uint index,
                            {},
                            {{}},
                            {{}}
                        ) {{{{
                            {}
                        }}}}\",
                        kernel_args.join(\",\n\"),
                        gating_vars_args.join(\",\n\"),
                    );

                    {}

                    {}

                    (
                        vec![{}, {}, {}],
                        update_function,
                    )
                }}",
                generate_kernel_args(&self.vars).iter()
                    .map(|i| format!("String::from(\"{}\")", i)).collect::<Vec<_>>().join(",\n"),
                generate_gating_vars_kernel_args(self.gating_vars.as_ref().unwrap()).join(",\n"),
                if kernel_on_iteration.contains("$update") {
                    update_gating_vars_func
                } else {
                    ""
                },
                self.type_name.generate(),
                if self.get_use_timestep() {
                    "__global float *current_voltage,\n__global float *dt,\n__global float *current"
                } else {
                    "__global float *current_voltage,\n__global float *current"
                },
                kernel_on_iteration,
                generate_gating_vars_replacements(self.gating_vars.as_ref().unwrap()).join("\n"),
                if kernel_on_iteration.contains("$update") {
                    generate_gating_vars_update_replacements(self.gating_vars.as_ref().unwrap())
                } else {
                    String::from("")
                },
                if self.get_use_timestep() {
                    "String::from(\"current_voltage\"), String::from(\"ion_channel$dt\"), String::from(\"ion_channel$current\")"
                } else {
                    "String::from(\"current_voltage\"), String::from(\"ion_channel$current\")"
                },
                generate_gpu_ion_channel_attributes_vec_no_types(&self.vars).join(", "),
                gating_vars_attrs_no_types.join(", "),
            )
        };

        let mut imports = vec![
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::write_buffer;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::read_and_set_buffer;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::AvailableBufferType;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::BufferType;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::BufferGPU;"),
            String::from("use spiking_neural_networks::error::GPUError;"),
            if self.get_use_timestep() { 
                String::from("use spiking_neural_networks::neuron::ion_channels::IonChannelGPU;")
            } else {
                String::from("use spiking_neural_networks::neuron::ion_channels::TimestepIndependentIonChannelGPU;")
            },
            String::from("use opencl3::command_queue::CommandQueue;"),
            String::from("use opencl3::types::CL_NON_BLOCKING;"),
            String::from("use opencl3::types::CL_BLOCKING;"),
            String::from("use opencl3::types::cl_float;"),
            String::from("use opencl3::types::cl_uint;"),
            String::from("use opencl3::memory::Buffer;"),
            String::from("use opencl3::memory::CL_MEM_READ_WRITE;"),
            String::from("use opencl3::context::Context;"),
            String::from("use std::collections::HashSet;"),
            String::from("use std::collections::HashMap;"),
            String::from("use std::ptr;"),
        ];

        if !gating_vars_attrs.is_empty() {
            imports.push(
                String::from("use spiking_neural_networks::neuron::iterate_and_spike::generate_unique_prefix;")
            );
        }

        (
            imports,
            format!(
                "impl {} for {} {{
                    {}
                    {}
                    {}
                    {}
                    {}
                    {}
                    {}
                }}",
                if self.get_use_timestep() { 
                    "IonChannelGPU" 
                } else {
                    "TimestepIndependentIonChannelGPU"
                },
                self.type_name.generate(),
                get_all_attrs_as_vec,
                get_all_attrs,
                get_attribute,
                set_attribute,
                convert_to_cpu,
                convert_to_gpu,
                update_function,
            )
        )
    }

    #[cfg(feature = "py")]
    fn to_pyo3_code(&self) -> (Vec<String>, String) {
        // determine fields (including default fields)
        // determine methods to impl
        // add getter setters to neuron that can modify ion channels

        let struct_def = format!(
            "#[pyclass]
            #[pyo3(name = \"{}\")]
            #[derive(Debug, Clone, Copy)]
            pub struct Py{} {{
                ion_channel: {}
            }}",
            self.type_name.generate(),
            self.type_name.generate(),
            self.type_name.generate(),
        );

        let update_current = if self.get_use_timestep() {
            "fn update_current(&mut self, voltage: f32, dt: f32) { self.ion_channel.update_current(voltage, dt) }"
        } else {
            "fn update_current(&mut self, voltage: f32) { self.ion_channel.update_current(voltage) }"
        };

        // if basic gating variable used
        // add definition of basic gating variable to imports
        // and import basic gating variable from snns package too

        let basic_gating_variable_def = "#[pyclass]
            #[pyo3(name = \"BasicGatingVariable\")]
            #[derive(Debug, Clone, Copy)]
            pub struct PyBasicGatingVariable {
                gating_variable: BasicGatingVariable
            }

            #[pymethods]
            impl PyBasicGatingVariable {
                #[new]
                #[pyo3(signature = (alpha=0., beta=0., state=0.))]
                fn new(alpha: f32, beta: f32, state: f32) -> Self {
                    PyBasicGatingVariable { 
                        gating_variable: BasicGatingVariable { alpha, beta, state }
                    } 
                }
                fn __repr__(&self) -> PyResult<String> { Ok(format!(\"{:#?}\", self.gating_variable)) }
                fn init_state(&mut self) { self.gating_variable.init_state(); }
                fn update(&mut self, dt: f32) { self.gating_variable.update(dt); }
                #[getter]
                fn get_alpha(&self) -> f32 {
                    self.gating_variable.alpha
                }
                #[setter]
                fn set_alpha(&mut self, new_param: f32) {
                    self.gating_variable.alpha = new_param;
                }
                #[getter]
                fn get_beta(&self) -> f32 {
                    self.gating_variable.beta
                }
                #[setter]
                fn set_beta(&mut self, new_param: f32) {
                    self.gating_variable.beta = new_param;
                }
                #[getter]
                fn get_state(&self) -> f32 {
                    self.gating_variable.state
                }
                #[setter]
                fn set_state(&mut self, new_param: f32) {
                    self.gating_variable.state = new_param;
                }
            }
        ";

        let gating_vars = match &self.gating_vars {
            Some(Ast::GatingVariables(variables)) => variables.iter()
                .map(|i| format!(
                    "fn get_{}(&self) -> PyBasicGatingVariable {{ 
                        PyBasicGatingVariable {{ gating_variable: self.ion_channel.{} }}
                    }}

                    fn set_{}(&mut self, new_param: PyBasicGatingVariable) {{ 
                        self.ion_channel.{} = new_param.gating_variable; 
                    }}",
                    i,
                    i,
                    i,
                    i,
                ))
                .collect(),
            None => vec![],
            _ => unreachable!()
        };

        let py_impl = format!(
            "#[pymethods]
            impl Py{} {{
                #[new]
                #[pyo3(signature = (current=0., {}, {}))]
                fn new(current: f32, {}, {}) -> Self {{ 
                    Py{} {{ 
                        ion_channel: {} {{
                            current,
                            {},
                            {}
                        }}
                    }} 
                }}
                fn __repr__(&self) -> PyResult<String> {{ Ok(format!(\"{{:#?}}\", self.ion_channel)) }}
                {}
                {}
                {}
                {}
            }}",
            self.type_name.generate(),
            generate_fields_as_fn_new_args(&self.vars).join(", "),
            generate_py_basic_gating_vars_as_fn_new_args(&self.gating_vars.as_ref()
                .unwrap_or(&Ast::GatingVariables(vec![]))
            ).join(", "),
            generate_fields_as_immutable_args(&self.vars).join(", "),
            generate_fields_as_py_basic_gating_vars_args(&self.gating_vars.as_ref()
                .unwrap_or(&Ast::GatingVariables(vec![]))
            ).join(","),
            self.type_name.generate(),
            self.type_name.generate(),
            generate_fields_as_names(&self.vars).join(",\n"),
            generate_py_basic_gating_vars_as_args_in_ion_channel(&self.gating_vars.as_ref()
                .unwrap_or(&Ast::GatingVariables(vec![]))
            ).join(",\n"),
            generate_vars_as_getter_setters("ion_channel", &self.vars).join("\n"),
            generate_py_getter_and_setters("ion_channel", "current", "f32"),
            gating_vars.join("\n"),
            update_current,
        );

        let mut imports = vec![String::from("use pyo3::prelude::*;")];

        if !gating_vars.is_empty() {
            imports.push(String::from(basic_gating_variable_def));
        }

        (
            imports,
            format!(
                "{}
                {}",
                struct_def,
                py_impl
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

fn parse_kinetics(pairs: Pairs<'_, Rule>) -> Result<(Ast, Ast, Ast)> {
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

    Ok((type_name, vars, on_iteration))
}

fn generate_neurotransmitter_kinetics(pairs: Pairs<Rule>) -> Result<NeurotransmitterKineticsDefinition> {
    let (type_name, vars, on_iteration) = parse_kinetics(pairs)?;
    
    Ok(NeurotransmitterKineticsDefinition { type_name, vars, on_iteration })    
}

#[cfg(feature="gpu")] 
fn generate_non_kernel_gpu_args(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    let type_name = match i {
                        Ast::VariableAssignment { value, .. } => match value {
                            NumOrBool::Number(_) => "float",
                            NumOrBool::Bool(_) => "uint",
                        },
                        _ => unreachable!(),
                    };

                    format!("{} {}", type_name, var_name)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature="gpu")] 
fn generate_non_kernel_gpu_on_iteration(on_iteration: &Ast) -> String {
    let on_iteration_assignments = on_iteration.generate_non_kernel_gpu();

    let changes = match on_iteration {
        Ast::OnIteration(assignments) => {
            let mut assignments_strings = vec![];

            for i in assignments {
                if let Ast::DiffEqAssignment { name, .. } =  i {
                    let change_string = if name == "v" {
                        "current_voltage += dv;".to_string()
                    } else {
                        format!("{} += d{};", name, name)
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

#[cfg(feature="gpu")] 
fn generate_gpu_matching<F>(vars: &Ast, format_type: F) -> Vec<String> 
where
    F: Fn(&str, &str) -> String,
{
    match vars {
        Ast::VariablesAssignments(variables) => {
            variables
                .iter()
                .map(|i| {
                    let var_name = match i {
                        Ast::VariableAssignment { name, .. } => name,
                        _ => unreachable!(),
                    };

                    let type_name = match i {
                        Ast::VariableAssignment { value, .. } => match value {
                            NumOrBool::Number(_) => "Float",
                            NumOrBool::Bool(_) => "UInt",
                        },
                        _ => unreachable!(),
                    };

                    format_type(var_name, type_name)
                })
                .collect::<Vec<String>>()
        },
        _ => unreachable!()
    }
}

#[cfg(feature="gpu")] 
fn generate_gpu_neurotransmitters_attribute_matching(vars: &Ast) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, type_name| { 
            format!(
                r#""neurotransmitters${}" => Some(BufferType::{}(self.{}))"#,
                var_name,
                type_name,
                var_name,
            )
        }
    )
}

#[cfg(feature="gpu")] 
fn generate_gpu_ion_channel_attribute_matching(vars: &Ast) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, type_name| { 
            format!(
                r#""ion_channel${}" => Some(BufferType::{}(self.{}))"#,
                var_name,
                type_name,
                var_name,
            )
        }
    )
}

#[cfg(feature="gpu")]
fn generate_gpu_gating_vars_attribute_matching(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::GatingVariables(variables) => {
            variables.iter()
                .map(|i|
                    format!(
                        "\"ion_channel${}$alpha\" => Some(BufferType::Float(self.{}.alpha)),
                        \"ion_channel${}$beta\" => Some(BufferType::Float(self.{}.beta)),
                        \"ion_channel${}$state\" => Some(BufferType::Float(self.{}.state))",
                        i,
                        i,
                        i,
                        i,
                        i,
                        i,
                    ) 
                )
                .collect()
        },
        _ => unreachable!()
    }
}

#[cfg(feature="gpu")] 
fn generate_gpu_neurotransmitters_attribute_setting(vars: &Ast) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, type_name| { 
            format!(
                r#""neurotransmitters${}" => self.{} = match value {{ 
                    BufferType::{}(nested_val) => nested_val,
                    _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),
                }}   
                "#,
                var_name,
                var_name,
                type_name,
            )
        }
    )
}

#[cfg(feature="gpu")] 
fn generate_gpu_ion_channel_attribute_setting(vars: &Ast) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, type_name| { 
            format!(
                r#""ion_channel${}" => self.{} = match value {{ 
                    BufferType::{}(nested_val) => nested_val,
                    _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),
                }}   
                "#,
                var_name,
                var_name,
                type_name,
            )
        }
    )
}

#[cfg(feature="gpu")] 
fn generate_gpu_gating_vars_attribute_setting(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::GatingVariables(variables) => {
            variables.iter()
                .map(|i|
                    format!(
                        r#""ion_channel${}$alpha" => self.{}.alpha = match value {{ 
                            BufferType::Float(nested_val) => nested_val,
                            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),
                        }},
                        "ion_channel${}$beta" => self.{}.beta = match value {{ 
                            BufferType::Float(nested_val) => nested_val,
                            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),
                        }},
                        "ion_channel${}$state" => self.{}.state = match value {{ 
                            BufferType::Float(nested_val) => nested_val,
                            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),
                        }}  
                        "#,
                        i,
                        i,
                        i,
                        i,
                        i,
                        i,
                    )
                )
                .collect()
        },
        _ => unreachable!(),
    }
}

#[cfg(feature="gpu")] 
fn generate_gpu_neurotransmitters_attributes_vec(vars: &Ast) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, type_name| { 
            format!(
                r#"(String::from("neurotransmitters${}"), AvailableBufferType::{})"#,
                var_name,
                type_name,
            )
        }
    )
}

#[cfg(feature="gpu")] 
fn generate_gpu_neurotransmitters_attributes_vec_no_types(vars: &Ast) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, _| { 
            format!(
                r#"(String::from("neurotransmitters${}"))"#,
                var_name,
            )
        }
    )
}

#[cfg(feature="gpu")] 
fn generate_gpu_ion_channel_attributes_vec_no_types(vars: &Ast) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, _| { 
            format!(
                r#"(String::from("ion_channel${}"))"#,
                var_name,
            )
        }
    )
}

#[cfg(feature="gpu")] 
fn generate_gpu_gating_vars_attributes_vec_no_types(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::GatingVariables(variables) => {
            variables.iter()
                .map(|i|
                    format!(
                        r#"(String::from("ion_channel${}$alpha")), 
                        (String::from("ion_channel${}$beta")),
                        (String::from("ion_channel${}$state"))"#,
                        i,
                        i,
                        i,
                    )
                )
                .collect()
        },
        _ => unreachable!()
    }
}

#[cfg(feature = "gpu")]
fn generate_gpu_receptors_attribute_matching(vars: &Ast, prefix: &str) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, type_name| { 
            format!(
                r#""receptors${}{}" => Some(BufferType::{}(self.{}))"#,
                prefix,
                var_name,
                type_name,
                var_name,
            )
        }
    )
}

#[cfg(feature = "gpu")]
fn generate_gpu_receptors_attribute_matching_inner_receptor(vars: &Ast, receptor_type_name: String, neurotransmitter: String) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, type_name| { 
            format!(
                "\"receptors${}_{}\" => match &self.receptors.get(&{}NeurotransmitterType::{}) {{\nSome({}Type::{}(val)) => Some(BufferType::{}(val.{})),\n_ => None\n}}",
                neurotransmitter,
                var_name,
                receptor_type_name,
                neurotransmitter,
                receptor_type_name,
                neurotransmitter,
                type_name,
                var_name,
            )
        }
    )
}

#[cfg(feature = "gpu")]
fn generate_gpu_receptors_attribute_setting(vars: &Ast, prefix: &str) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, type_name| { 
            format!(
                r#""receptors${}{}" => self.{} = match value {{ 
                    BufferType::{}(nested_val) => nested_val,
                    _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),
                }}   
                "#,
                prefix,
                var_name,
                var_name,
                type_name,
            )
        }
    )
}

#[cfg(feature = "gpu")]
fn generate_gpu_receptors_attribute_setting_inner_receptor(vars: &Ast, receptor_type_name: String, neurotransmitter: String) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, type_name| { 
            format!(
                r#""receptors${}_{}" => match (self.receptors.get_mut(&{}NeurotransmitterType::{}), value) {{ 
                    (Some({}Type::{}(receptors)), BufferType::{}(nested_val)) => receptors.{} = nested_val,
                    _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid type")),
                }}   
                "#,
                neurotransmitter,
                var_name,
                receptor_type_name,
                neurotransmitter,
                receptor_type_name,
                neurotransmitter,
                type_name,
                var_name,
            )
        }
    )
}

#[cfg(feature = "gpu")]
fn generate_gpu_receptors_attributes_vec(vars: &Ast, prefix: &str) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, type_name| { 
            format!(
                r#"(String::from("receptors${}{}"), AvailableBufferType::{})"#,
                prefix,
                var_name,
                type_name,
            )
        }
    )
}

#[cfg(feature = "gpu")]
fn generate_gpu_receptors_attributes_vec_no_types(vars: &Ast, prefix: &str) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, _| { 
            format!(
                r#"(String::from("receptors${}{}"))"#,
                prefix,
                var_name,
            )
        }
    )
}

#[cfg(feature = "gpu")]
fn generate_gpu_ion_channel_attributes_vec(vars: &Ast) -> Vec<String> {
    generate_gpu_matching(
        vars, 
        |var_name, type_name| { 
            format!(
                r#"(String::from("ion_channel${}"), AvailableBufferType::{})"#,
                var_name,
                type_name,
            )
        }
    )
}

#[cfg(feature = "gpu")]
fn generate_gpu_gating_vars_attributes_vec(vars: &Ast) -> Vec<String> {
    match vars {
        Ast::GatingVariables(variables) => {
            variables.iter()
                .map(|i|
                    format!(
                        r#"(String::from("ion_channel${}$alpha"), AvailableBufferType::Float), 
                        (String::from("ion_channel${}$beta"), AvailableBufferType::Float), 
                        (String::from("ion_channel${}$state"), AvailableBufferType::Float)
                        "#,
                        i,
                        i,
                        i,
                    )
                )
                .collect()
        },
        _ => unreachable!()
    }
}

impl NeurotransmitterKineticsDefinition {
    fn to_code(&self) -> (Vec<String>, String) {
        let imports = vec![
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::CurrentVoltage;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::Timestep;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::IsSpiking;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::NeurotransmitterKinetics;"),
        ];

        let header = format!(
            "#[derive(Debug, Clone, Copy, PartialEq)]\npub struct {} {{", 
            self.type_name.generate(),
        );
        
        let mut fields = generate_fields(&self.vars);

        let t_field = String::from("pub t: f32");
        fields.push(t_field);

        let fields = format!("\t{},", fields.join(",\n\t"));

        let update_body = generate_on_iteration(&self.on_iteration);

        let update_body = replace_self_var(update_body, "is_spiking", "neuron.is_spiking()");
        let update_body = replace_self_var(update_body, "dt", "neuron.get_dt()");
        let update_body = replace_self_var(update_body, "current_voltage", "neuron.get_current_voltage()");

        let update_body = format!(
            "fn apply_t_change<T: CurrentVoltage + IsSpiking + Timestep>(&mut self, neuron: &T) {{\n{}\n}}",
            update_body
        );

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

        let get_t = "fn get_t(&self) -> f32 { self.t }";
        let set_t = "fn set_t(&mut self, t: f32) { self.t = t; }";

        (
            imports,
            format!(
                "{}\n{}\n}}\n\n{}\n\n{}\n{}\n\n{}\n\n{}\n}}\n", 
                header, 
                fields, 
                default_function,
                impl_header, 
                update_body, 
                get_t,
                set_t,
            )
        )
    }

    #[cfg(feature="gpu")] 
    fn to_gpu_code(&self) -> (Vec<String>, String) {
        let kinetics_function_header = format!(
            "float get_t(float current_voltage, uint is_spiking, float dt, float t, {}) {{", 
            generate_non_kernel_gpu_args(&self.vars).join(", "),
        );

        let kinetics_body = generate_non_kernel_gpu_on_iteration(&self.on_iteration);

        let kinetics_function = format!("{}\n{}\nreturn t;\n}}", kinetics_function_header, kinetics_body);

        let impl_header = format!("impl NeurotransmitterKineticsGPU for {} {{", self.type_name.generate());
        let get_attribute_header = "fn get_attribute(&self, value: &str) -> Option<BufferType> {";
        let get_attribute_body = format!(
            "match value {{ \"neurotransmitters$t\" => Some(BufferType::Float(self.t)),\n{},\n_ => None }}", 
            generate_gpu_neurotransmitters_attribute_matching(&self.vars).join(",\n")
        );
        
        let get_function = format!("{}\n{}}}", get_attribute_header, get_attribute_body);

        let set_attribute_header = "fn set_attribute(&mut self, attribute: &str, value: BufferType) -> Result<(), std::io::Error> {";
        let set_t_attribute = "\"neurotransmitters$t\" => self.t = match value {
                BufferType::Float(nested_val) => nested_val,
                _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, \"Invalid type\")),
            }";
        let set_attribute_body = format!(
            "match attribute {{ {},\n{},\n_ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, \"Invalid attribute\")) }};\nOk(())",
            set_t_attribute,
            generate_gpu_neurotransmitters_attribute_setting(&self.vars).join(",\n")
        );

        let set_function = format!("{}\n{}\n}}", set_attribute_header, set_attribute_body);

        let vector_return_header = "fn get_attribute_names_as_vector() -> Vec<(String, AvailableBufferType)> {";
        let vector_return_function = format!(
            "{}vec![(String::from(\"neurotransmitters$t\"), AvailableBufferType::Float), {}\n]\n}}", 
            vector_return_header,
            generate_gpu_neurotransmitters_attributes_vec(&self.vars).join(",\n"),
        );

        let attribute_names_functions = r#"
            fn get_attribute_names() -> HashSet<(String, AvailableBufferType)> {
                HashSet::from_iter(
                    Self::get_attribute_names_as_vector()
                )
            }

            fn get_attribute_names_ordered() -> BTreeSet<(String, AvailableBufferType)> {
                Self::get_attribute_names_as_vector().into_iter().collect()
            }
            "#;

        let mandatory_args = r#"String::from("current_voltage"), String::from("is_spiking"), String::from("dt")"#;

        let get_update_function_header = "fn get_update_function() -> ((Vec<String>, Vec<String>), String) {";
        let get_update_function = format!(
            "{}\n((vec![{}],\nvec![String::from(\"neurotransmitters$t\"), {}]),\nString::from(\"{}\"))\n}}",
            get_update_function_header,
            mandatory_args,
            generate_gpu_neurotransmitters_attributes_vec_no_types(&self.vars).join(","),
            kinetics_function,
        );

        // format fns together and get imports

        let imports = vec![
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::NeurotransmitterKineticsGPU;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::BufferType;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::AvailableBufferType;"),
            String::from("use std::collections::HashSet;"),
            String::from("use std::collections::BTreeSet;"),
        ];

        (
            imports,
            format!("
                    {}
                    {}
                    {}
                    {}
                    {}
                    {}
                }} 
                ",
                impl_header,
                get_function,
                set_function,
                vector_return_function,
                attribute_names_functions,
                get_update_function,
            )
        )
    }

    #[cfg(feature = "py")]
    fn to_pyo3_code(&self) -> (Vec<String>, String) {
        let struct_def = format!("
            #[pyclass]
            #[pyo3(name = \"{}\")]
            #[derive(Clone, Copy)]
            pub struct Py{} {{
                neurotransmitter: {},
            }}",
            self.type_name.generate(),
            self.type_name.generate(),
            self.type_name.generate(),
        );

        let mandatory_vars = [("t", "f32")];

        let mandatory_getter_and_setters: Vec<String> = mandatory_vars.iter()
            .map(|(i, j)| generate_py_getter_and_setters("neurotransmitter", i, j))
            .collect();

        let mut basic_getter_setters = generate_vars_as_getter_setters("neurotransmitter", &self.vars);  
        basic_getter_setters.extend(mandatory_getter_and_setters);

        let apply_t_changes_func = "fn apply_t_change(&mut self, voltage: f32, is_spiking: bool, dt: f32) {
            self.neurotransmitter.apply_t_change(&NeurotransmittersIntermediate { current_voltage: voltage, is_spiking, dt });
        }";

        let constructor = format!(
            "#[new]
            #[pyo3(signature = (t=0., {}))]
            fn new(t: f32, {}) -> Self {{
                Py{} {{ 
                    neurotransmitter: {} {{
                        t,
                        {}
                    }}
                }}
            }}",
            generate_fields_as_fn_new_args(&self.vars).join(", "),
            generate_fields_as_immutable_args(&self.vars).join(", "),
            self.type_name.generate(),
            self.type_name.generate(),
            generate_fields_as_names(&self.vars).join(",\n"),
        );

        let repr = r#"fn __repr__(&self) -> PyResult<String> { Ok(format!("{:#?}", self.neurotransmitter)) }"#;

        let imports = vec![
            String::from("use pyo3::prelude::*;"), 
            String::from("use spiking_neural_networks::neuron::intermediate_delegate::NeurotransmittersIntermediate;"),
        ];

        let py_impl = format!("
                #[pymethods]
                impl Py{} {{
                    {}
                    {}
                    {}
                    {}
                }}
            ",
            self.type_name.generate(),
            constructor,
            repr,
            basic_getter_setters.join("\n"),
            apply_t_changes_func,
        );

        (
            imports,
            format!(
                "{}
                {}",
                struct_def,
                py_impl,
            )
        )
    }
}

struct ReceptorKineticsDefinition {
    type_name: Ast,
    vars: Ast,
    on_iteration: Ast,
}

fn generate_receptor_kinetics(pairs: Pairs<Rule>) -> Result<ReceptorKineticsDefinition> {
    let (type_name, vars, on_iteration) = parse_kinetics(pairs)?;
    
    Ok(ReceptorKineticsDefinition { type_name, vars, on_iteration })    
}

impl ReceptorKineticsDefinition {
    fn to_code(&self) -> (Vec<String>, String) {
        let imports = vec![
            String::from(
                "use spiking_neural_networks::neuron::iterate_and_spike::ReceptorKinetics;"
            )
        ];

        let header = format!(
            "#[derive(Debug, Clone, Copy, PartialEq)]\npub struct {} {{", 
            self.type_name.generate(),
        );
        
        let mut fields = generate_fields(&self.vars);

        let t_field = String::from("pub r: f32");
        fields.push(t_field);

        let fields = format!("\t{},", fields.join(",\n\t"));

        let update_body = generate_on_iteration(&self.on_iteration);

        let update_body = replace_self_var(update_body, "t", "t");
        let update_body = replace_self_var(update_body, "dt", "neuron.get_dt()");

        let update_body = format!(
            "fn apply_r_change(&mut self, t: f32, dt: f32) {{\n{}\n}}",
            update_body
        );

        let mut defaults = generate_defaults(&self.vars);

        defaults.push(String::from("r: 0."));

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

        let impl_header = format!("impl ReceptorKinetics for {} {{", self.type_name.generate());

        let get_r = "fn get_r(&self) -> f32 { self.r }";
        let set_r = "fn set_r(&mut self, r: f32) { self.r = r; }";

        (
            imports,
            format!(
                "{}\n{}\n}}\n\n{}\n\n{}\n{}\n\n{}\n\n{}\n}}\n", 
                header, 
                fields, 
                default_function,
                impl_header, 
                update_body, 
                get_r,
                set_r,
            )
        )
    }

    #[cfg(feature = "gpu")]
    fn to_gpu_code(&self) -> (Vec<String>, String) {
        let kinetics_function_header = format!(
            "float get_r(float t, float dt, float r, {}) {{", 
            generate_non_kernel_gpu_args(&self.vars).join(", "),
        );

        let kinetics_body = generate_non_kernel_gpu_on_iteration(&self.on_iteration);

        let kinetics_function = format!("{}\n{}\nreturn r;\n}}", kinetics_function_header, kinetics_body);

        let impl_header = format!("impl ReceptorKineticsGPU for {} {{", self.type_name.generate());
        let get_attribute_header = "fn get_attribute(&self, value: &str) -> Option<BufferType> {";
        let get_attribute_body = format!(
            "match value {{ \"receptors$kinetics$r\" => Some(BufferType::Float(self.r)),\n{},\n_ => None }}", 
            generate_gpu_receptors_attribute_matching(&self.vars, "kinetics$").join(",\n")
        );
        
        let get_function = format!("{}\n{}}}", get_attribute_header, get_attribute_body);

        let set_attribute_header = "fn set_attribute(&mut self, attribute: &str, value: BufferType) -> Result<(), std::io::Error> {";
        let set_r_attribute = "\"receptors$kinetics$r\" => self.r = match value {
                BufferType::Float(nested_val) => nested_val,
                _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, \"Invalid type\")),
            }";
        let set_attribute_body = format!(
            "match attribute {{ {},\n{},\n_ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, \"Invalid attribute\")) }};\nOk(())",
            set_r_attribute,
            generate_gpu_receptors_attribute_setting(&self.vars, "kinetics$").join(",\n")
        );

        let set_function = format!("{}\n{}\n}}", set_attribute_header, set_attribute_body);

        let vector_return_header = "fn get_attribute_names() -> HashSet<(String, AvailableBufferType)> {";
        let vector_return_function = format!(
            "{}HashSet::from([(String::from(\"receptors$kinetics$r\"), AvailableBufferType::Float), {}\n])\n}}", 
            vector_return_header,
            generate_gpu_receptors_attributes_vec(&self.vars, "kinetics$").join(",\n"),
        );

        let get_update_function_header = "fn get_update_function() -> (Vec<String>, String) {";
        let get_update_function = format!(
            "{}\n((\nvec![String::from(\"neurotransmitters$t\"), String::from(\"dt\"), String::from(\"receptors$kinetics$r\"), {}]),\nString::from(\"{}\"))\n}}",
            get_update_function_header,
            generate_gpu_receptors_attributes_vec_no_types(&self.vars, "kinetics$").join(","),
            kinetics_function,
        );

        let imports = vec![
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::ReceptorKineticsGPU;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::BufferType;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::AvailableBufferType;"),
            String::from("use std::collections::HashSet;"),
        ];

        (
            imports,
            format!("
                    {}
                    {}
                    {}
                    {}
                    {}
                }} 
                ",
                impl_header,
                get_function,
                set_function,
                vector_return_function,
                get_update_function,
            )
        )
    }

    #[cfg(feature = "py")]
    fn to_pyo3_code(&self) -> (Vec<String>, String) {
        let struct_def = format!("
            #[pyclass]
            #[pyo3(name = \"{}\")]
            #[derive(Clone, Copy)]
            pub struct Py{} {{
                receptor: {},
            }}",
            self.type_name.generate(),
            self.type_name.generate(),
            self.type_name.generate(),
        );

        let mandatory_vars = [("r", "f32")];

        let mandatory_getter_and_setters: Vec<String> = mandatory_vars.iter()
            .map(|(i, j)| generate_py_getter_and_setters("receptor", i, j))
            .collect();

        let mut basic_getter_setters = generate_vars_as_getter_setters("receptor", &self.vars);  
        basic_getter_setters.extend(mandatory_getter_and_setters);

        let apply_r_changes_func = "fn apply_r_change(&mut self, t: f32, dt: f32) {
            self.receptor.apply_r_change(t, dt);
        }";

        let constructor = format!(
            "#[new]
            #[pyo3(signature = (r=0., {}))]
            fn new(r: f32, {}) -> Self {{
                Py{} {{ 
                    receptor: {} {{
                        r,
                        {}
                    }}
                }}
            }}",
            generate_fields_as_fn_new_args(&self.vars).join(", "),
            generate_fields_as_immutable_args(&self.vars).join(", "),
            self.type_name.generate(),
            self.type_name.generate(),
            generate_fields_as_names(&self.vars).join(",\n"),
        );

        let repr = r#"fn __repr__(&self) -> PyResult<String> { Ok(format!("{:#?}", self.receptor)) }"#;

        let imports = vec![String::from("use pyo3::prelude::*;")];

        let py_impl = format!("
                #[pymethods]
                impl Py{} {{
                    {}
                    {}
                    {}
                    {}
                }}
            ",
            self.type_name.generate(),
            constructor,
            repr,
            basic_getter_setters.join("\n"),
            apply_r_changes_func,
        );

        (
            imports,
            format!(
                "{}
                {}",
                struct_def,
                py_impl,
            )
        )
    }
}

struct ReceptorsDefinition {
    type_name: Ast,
    default_kinetics: Option<Ast>,
    top_level_vars: Option<Ast>,
    blocks: Vec<(Ast, Ast, Ast, Ast)>,
}

fn parse_single_kinetics_definition(pair: Pair<'_, Rule>) -> (String, Ast) {
    (
        String::from("single_kinetics"), 
        Ast::SingleKineticsDefinition(
            String::from(pair.into_inner().next().unwrap().as_str())
        )
    )
}

fn parse_receptor_vars_def(pair: Pair<'_, Rule>) -> (String, Ast) {
    let inner_rules = pair.into_inner();

    let assignments: Vec<Ast> = inner_rules 
        .map(|i| {
            Ast::Name(String::from(i.as_str()))
        })
        .collect(); 


    (
        String::from("receptors_var_def"),
        Ast::VariablesAssignments(assignments),
    )
}

fn generate_receptors(pairs: Pairs<Rule>) -> Result<ReceptorsDefinition> {
    // hashmap for top level vars and type name and nested hashmap for tuples in generate function 

    let mut definitions: HashMap<String, Ast> = HashMap::new();
    let mut blocks: Vec<(Ast, Ast, Ast, Ast)> = vec![];

    for pair in pairs {
        match pair.as_rule() {
            Rule::type_def => {
                let (key, current_ast) = parse_type_definition(pair);

                if definitions.contains_key(&key) {
                    return Err(
                        Error::new(
                            ErrorKind::InvalidInput, format!("Duplicate definition found: {}", key),
                        )
                    )
                }

                definitions.insert(key, current_ast);
            },
            Rule::vars_with_default_def => {
                let (key, current_ast) = parse_vars_with_default(pair);

                if definitions.contains_key(&key) {
                    return Err(
                        Error::new(
                            ErrorKind::InvalidInput, format!("Duplicate definition found: {}", key),
                        )
                    )
                }

                definitions.insert(key, current_ast);
            },
            Rule::single_kinetics_def => {
                let (key, current_ast) = parse_single_kinetics_definition(pair);

                if definitions.contains_key(&key) {
                    return Err(
                        Error::new(
                            ErrorKind::InvalidInput, format!("Duplicate definition found: {}", key),
                        )
                    )
                }

                definitions.insert(key, current_ast);
            },
            Rule::receptor_block => {
                let mut new_block: HashMap<String, Ast> = HashMap::new();

                for inner_pair in pair.into_inner() {
                    let (key, current_ast)  = match inner_pair.as_rule() {
                        Rule::neurotransmitter_def => (
                            String::from("neurotransmitter"), 
                            Ast::TypeDefinition(
                                String::from(inner_pair.into_inner().next().unwrap().as_str())
                            )
                        ),
                        Rule::vars_with_default_def => {
                            parse_vars_with_default(inner_pair)
                        },
                        Rule::on_iteration_def => {
                            parse_on_iteration(inner_pair)
                        },
                        Rule::receptor_vars_def => {
                            parse_receptor_vars_def(inner_pair)
                        }
                        definition => unreachable!("Unexpected definition: {:#?}", definition)
                    };

                    new_block.insert(key, current_ast);
                }

                let neurotransmitter_block = match new_block.remove("neurotransmitter") {
                    Some(val) => val,
                    None => {
                        return Err(Error::new(ErrorKind::InvalidData, "Missing neurotransmitter definition"))
                    }
                };
                let vars_block = match new_block.remove("vars") {
                    Some(val) => val,
                    None => {
                        return Err(Error::new(ErrorKind::InvalidData, "Missing variables definition"))
                    }
                };
                let on_iteration_block = match new_block.remove("on_iteration") {
                    Some(val) => val,
                    None => {
                        return Err(Error::new(ErrorKind::InvalidData, "Missing on iteration definition"))
                    }
                };
                let receptors_var_def_block = match new_block.remove("receptors_var_def") {
                    Some(val) => val,
                    None => {
                        Ast::VariablesAssignments(vec![Ast::Name(String::from("r"))])
                    }
                };

                blocks.push(
                    (
                        neurotransmitter_block,
                        vars_block,
                        on_iteration_block,
                        receptors_var_def_block,
                    )
                );
            }
            definition => unreachable!("Unexpected definition: {:#?}", definition)
        };
    }

    let type_name = definitions.remove("type").ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "Type name definition expected")
    })?;
    let default_kinetics = definitions.remove("single_kinetics");
    let vars = definitions.remove("vars");

    Ok(
        ReceptorsDefinition { 
            type_name,
            default_kinetics,
            top_level_vars: vars, 
            blocks,
        }
    )
}

#[cfg(feature = "gpu")]
fn generate_receptor_setting_inner_kinetics(neurotransmitters_to_receptor_vars: &HashMap<String, Vec<String>>, receptor_type_name: &str) -> Vec<String> {
    let mut output = vec![];

    for (neuro, names) in neurotransmitters_to_receptor_vars.iter() {
        for name in names.iter() {
            output.push(format!(
                "\"{}${}\" => match self.receptors.get_mut(&{}NeurotransmitterType::{}) {{ 
                    Some({}Type::{}(inner)) => inner.{}.set_attribute(&stripped, value)?,
                    _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, \"Invalid attribute\")), 
                }}",
                neuro,
                &name[5..],
                receptor_type_name,
                neuro,
                receptor_type_name,
                neuro,
                &name[5..],
            ));
        }
    }

    output
}

#[cfg(feature = "gpu")]
fn generate_receptor_matching_inner_kinetics(neurotransmitters_to_receptor_vars: &HashMap<String, Vec<String>>, receptor_type_name: &str) -> Vec<String> {
    let mut output = vec![];

    for (neuro, names) in neurotransmitters_to_receptor_vars.iter() {
        for name in names.iter() {
            output.push(format!(
                "\"{}${}\" => match self.receptors.get(&{}NeurotransmitterType::{}) {{ 
                    Some({}Type::{}(inner)) => inner.{}.get_attribute(&stripped),
                    _ => None, 
                }}",
                neuro,
                &name[5..],
                receptor_type_name,
                neuro,
                receptor_type_name,
                neuro,
                &name[5..],
            ));
        }
    }

    output
}

impl ReceptorsDefinition {
    // generate neurotransmitter type from the blocks
    // for each neurotransmitter type create a receptors enum with structs in enum
    // structs in enum get associated vars
    // each struct also calculates a current
    // then move to metabotropic
    // should have simple way to edit gmax when receptors struct is initialized
    fn to_code(&self) -> (Vec<String>, String) {
        let mut imports = vec![
            String::from("use std::collections::HashMap;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::ReceptorKinetics;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::NeurotransmitterConcentrations;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::Receptors;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::NeurotransmitterType;"),
            String::from("use spiking_neural_networks::error::ReceptorNeurotransmitterError;"),
        ];

        let neurotransmitters_name = format!("{}NeurotransmitterType", self.type_name.generate());
        let neurotransmitter_types: Vec<String> = self.blocks.iter()
            .map(|i| i.0.generate())
            .collect();

        let impl_neurotransmitter_type = format!("impl NeurotransmitterType for {} {{}}", neurotransmitters_name);

        let neurotransmitters_definiton = format!(
            "#[derive(Hash, Debug, Eq, PartialEq, PartialOrd, Ord, Clone, Copy)]\npub enum {} {{\n{}\n}}\n{}", 
            neurotransmitters_name,
            neurotransmitter_types.join(",\n"),
            impl_neurotransmitter_type,
        );

        let has_top_level_vars = &self.top_level_vars.is_some();

        let mut receptors = vec![];
        let mut receptor_names = vec![];
        let mut has_current = vec![];

        for (type_name, vars_def, on_iteration, receptor_vars) in &self.blocks {
            receptor_names.push(type_name.generate());

            let mut vars = generate_fields(vars_def);
            let mut defaults = generate_defaults(vars_def);

            if let Ast::VariablesAssignments(inner_vars) = receptor_vars {
                for var in inner_vars {
                    vars.push(format!("pub {}: T", var.generate().replace("self.", "")));
                    defaults.push(format!("{}: T::default()", var.generate().replace("self.", "")));
                }
            }

            if vars.contains(&String::from("pub current: f32")) {
                has_current.push(type_name.generate());
            }

            let struct_def = format!(
                "#[derive(Debug, Clone, PartialEq)]\npub struct {}Receptor<T: ReceptorKinetics> {{\n{}\n}}", 
                type_name.generate(),
                vars.join(",\n")
            );

            let struct_defaults = format!(
                "impl<T: ReceptorKinetics> Default for {}Receptor<T> {{\nfn default() -> Self {{\n{}Receptor {{\n{}\n}}}}}}",
                type_name.generate(),
                type_name.generate(),
                defaults.join(",\n"),
            );

            let mut iterate_block = on_iteration.generate();
            if *has_top_level_vars {
                for i in generate_fields_as_names(self.top_level_vars.as_ref().unwrap()) {
                    iterate_block = replace_self_var(iterate_block, &i, &format!("*{}", i));
                }
            }

            if let Ast::VariablesAssignments(inner_vars) = receptor_vars {
                for var in inner_vars {
                    iterate_block = replace_self_var(
                        iterate_block, 
                        &var.generate().replace("self.", ""), 
                        &format!("{}.get_r()", var.generate()),
                    );
                }
            }
            
            iterate_block = replace_self_var(iterate_block, "current_voltage", "current_voltage");

            let mut apply_r_changes = vec![];
            if let Ast::VariablesAssignments(inner_vars) = receptor_vars {
                for var in inner_vars {
                    apply_r_changes.push(format!("{}.apply_r_change(t, dt);", var.generate()))
                }
            }

            let receptor_impl = if !has_top_level_vars {
                format!(
                    "impl<T: ReceptorKinetics> {}Receptor<T> {{ 
                        fn apply_r_change(&mut self, t: f32, dt: f32) {{ {} }}
                        fn iterate(&mut self, current_voltage: f32, dt: f32) {{ {} }}
                    }}
                    ",
                    type_name.generate(),
                    apply_r_changes.join("\n"),
                    iterate_block,
                )
            } else {
                format!(
                    "impl<T: ReceptorKinetics> {}Receptor<T> {{ 
                        fn apply_r_change(&mut self, t: f32, dt: f32) {{ {} }}
                        fn iterate(&mut self, current_voltage: f32, dt: f32, {}) {{ {} }}
                    }}
                    ",
                    type_name.generate(),
                    apply_r_changes.join("\n"),
                    generate_fields_as_args(self.top_level_vars.as_ref().unwrap()).join(", "),
                    iterate_block,
                )
            };

            receptors.push(
                format!(
                    "{}\n{}\n{}", 
                    struct_def, 
                    struct_defaults,
                    receptor_impl,
                )
            );
        }

        // when implementing receptors together
        // check if there is a current
        // if not ignore
        // otherwise add it to the currents to sum together ionotropically

        let receptor_enum = format!(
            "#[derive(Debug, Clone, PartialEq)]\npub enum {}Type<T: ReceptorKinetics> {{\n{}\n}}",
            self.type_name.generate(),
            receptor_names.iter()
                .map(|i| format!("{}({}Receptor<T>)", i, i))
                .collect::<Vec<String>>()
                .join("\n,")
        );

        let receptors_struct = if !has_top_level_vars {
            format!(
                "#[derive(Debug, Clone, PartialEq)]\npub struct {}<T: ReceptorKinetics> {{\nreceptors: HashMap<{}, {}Type<T>>\n}}", 
                self.type_name.generate(),
                neurotransmitters_name,
                self.type_name.generate(),
            )  
        } else {
            format!(
                "#[derive(Debug, Clone, PartialEq)]\npub struct {}<T: ReceptorKinetics> {{\n{},\nreceptors: HashMap<{}, {}Type<T>>\n}}", 
                self.type_name.generate(),
                generate_fields(self.top_level_vars.as_ref().unwrap()).join(",\n"),
                neurotransmitters_name,
                self.type_name.generate(),
            ) 
        };

        let update_receptor_kinetics = format!(
            "fn update_receptor_kinetics(&mut self, t: &NeurotransmitterConcentrations<{}>, dt: f32) {{
                t.iter()
                    .for_each(|(key, value)| {{
                        if let Some(receptor_type) = self.receptors.get_mut(key) {{
                            match receptor_type {{
                                {}
                            }}
                        }}
                    }});
            }}",
            neurotransmitters_name,
            receptor_names.iter()
                .map(|name| 
                    format!(
                        "{}Type::{}(receptor) => {{ receptor.apply_r_change(*value, dt); }}", 
                        self.type_name.generate(), 
                        name
                    )
                )
                .collect::<Vec<String>>()
                .join(",\n")
        );

        let check_receptor_neurotransmitter_type = neurotransmitter_types.iter().zip(receptor_names.iter())
            .map(|(i, j)| 
                format!("
                    if let {}Type::{}(_) = receptor_type {{
                        if neurotransmitter_type == {}::{} {{
                            is_valid = true;
                        }}
                    }}
                    ",
                    self.type_name.generate(),
                    j,
                    neurotransmitters_name,
                    i,
                )
            ).collect::<Vec<String>>();

        let insert = format!(
            "
                fn insert(&mut self, neurotransmitter_type: Self::N, receptor_type: Self::R) -> Result<(), ReceptorNeurotransmitterError> {{
                    let mut is_valid = false;

                    {}

                    if !is_valid {{
                        return Err(ReceptorNeurotransmitterError::MismatchedTypes);
                    }}

                    self.receptors.insert(neurotransmitter_type, receptor_type);

                    Ok(())
                }}
            ",
            check_receptor_neurotransmitter_type.join("\n"),
        );
 
        let receptors_default = if !has_top_level_vars {
            format!(
                "impl<T: ReceptorKinetics> Default for {}<T> {{\nfn default() -> Self {{{} {{ receptors: HashMap::new() }} }} }}",
                self.type_name.generate(),
                self.type_name.generate(),
            )
        } else {
            format!(
                "impl<T: ReceptorKinetics> Default for {}<T> {{\nfn default() -> Self {{{} {{ {},\nreceptors: HashMap::new() }} }} }}",
                self.type_name.generate(),
                self.type_name.generate(),
                generate_defaults(self.top_level_vars.as_ref().unwrap()).join(",\n"),
            )
        };

        if self.default_kinetics.is_none() {
            imports.push(
                String::from("use spiking_neural_networks::neuron::iterate_and_spike::ApproximateReceptor;")
            );
        }

        let default_impl = format!(
            "impl {}<{}> {{ pub fn default_impl() -> Self {{ {}::default() }} }}",
            self.type_name.generate(),
            self.default_kinetics.as_ref().unwrap_or(
                &Ast::SingleKineticsDefinition(String::from("ApproximateReceptor"))
            ).generate(),
            self.type_name.generate(),
        );

        if has_current.is_empty() {
            let receptors_impl = format!(
                "impl<T: ReceptorKinetics> Receptors for {}<T> {{
                    type T = T;
                    type N = {};
                    type R = {}Type<T>;
                    {}
                    fn get(&self, neurotransmitter_type: &Self::N) -> Option<&Self::R> {{\nself.receptors.get(neurotransmitter_type)\n}}
                    fn get_mut(&mut self, neurotransmitter_type: &Self::N) -> Option<&mut Self::R> {{\nself.receptors.get_mut(neurotransmitter_type)\n}}
                    fn len(&self) -> usize {{\nself.receptors.len()\n}}
                    fn is_empty(&self) -> bool {{\nself.receptors.is_empty()\n}}
                    fn remove(&mut self, neurotransmitter_type: &Self::N) -> Option<Self::R> {{ self.receptors.remove(&neurotransmitter_type) }}
                    {}
                }}
                ",
                self.type_name.generate(),
                neurotransmitters_name,
                self.type_name.generate(),
                update_receptor_kinetics,
                insert,
            );
            
            return (
                imports,
                format!(
                    "{}\n{}\n{}\n{}\n{}\n{}\n{}", 
                    neurotransmitters_definiton, 
                    receptors.join("\n"),
                    receptor_enum,
                    receptors_struct,
                    receptors_impl,
                    receptors_default,
                    default_impl,
                )
            );
        }

        imports.push(String::from("use spiking_neural_networks::neuron::iterate_and_spike::IonotropicReception;"));

        let set_receptor_currents = receptor_names.iter()
            .map(|name| 
                if !has_top_level_vars {
                    format!(
                        "if let Some(receptor_type) = self.receptors.get_mut(&{}::{}) {{
                            if let {}Type::{}(receptor) = receptor_type {{
                                receptor.iterate(current_voltage, dt);
                            }}
                        }}",
                        neurotransmitters_name,
                        name,
                        self.type_name.generate(),
                        name
                    )
                } else {
                    format!(
                        "if let Some(receptor_type) = self.receptors.get_mut(&{}::{}) {{
                            if let {}Type::{}(receptor) = receptor_type {{
                                receptor.iterate(current_voltage, dt, {});
                            }}
                        }}",
                        neurotransmitters_name,
                        name,
                        self.type_name.generate(),
                        name,
                        generate_fields_as_args_to_pass(self.top_level_vars.as_ref().unwrap())
                            .join(", "),
                    )
                }
            )
            .collect::<Vec<String>>()
            .join("\n");

        let get_receptor_currents = has_current.iter()
            .map(|name| 
                format!(
                    "if let Some(receptor_type) = self.receptors.get(&{}::{}) {{
                        if let {}Type::{}(receptor) = receptor_type {{
                            total += receptor.current;
                        }}
                    }}",
                    neurotransmitters_name,
                    name,
                    self.type_name.generate(),
                    name
                )
            )
            .collect::<Vec<String>>()
            .join("\n");

        let get_receptor_currents = format!(
            "let mut total = 0.;\n{};\ntotal * (dt / c_m)", 
            get_receptor_currents
        );

        let receptors_impl = format!(
            "impl<T: ReceptorKinetics> Receptors for {}<T> {{
                type T = T;
                type N = {};
                type R = {}Type<T>;
                {}
                fn get(&self, neurotransmitter_type: &Self::N) -> Option<&Self::R> {{\nself.receptors.get(neurotransmitter_type)\n}}
                fn get_mut(&mut self, neurotransmitter_type: &Self::N) -> Option<&mut Self::R> {{\nself.receptors.get_mut(neurotransmitter_type)\n}}
                fn len(&self) -> usize {{\nself.receptors.len()\n}}
                fn is_empty(&self) -> bool {{\nself.receptors.is_empty()\n}}
                fn remove(&mut self, neurotransmitter_type: &Self::N) -> Option<Self::R> {{ self.receptors.remove(&neurotransmitter_type) }}
                {}
            }}

            impl<T: ReceptorKinetics> IonotropicReception for {}<T> {{
                fn set_receptor_currents(&mut self, current_voltage: f32, dt: f32) {{\n{}\n}}
                fn get_receptor_currents(&self, dt: f32, c_m: f32) -> f32 {{\n{}\n}}
            }}
            ",
            self.type_name.generate(),
            neurotransmitters_name,
            self.type_name.generate(),
            update_receptor_kinetics,
            insert,
            self.type_name.generate(),
            set_receptor_currents,
            get_receptor_currents,
        );

        (
            imports, 
            format!(
                "{}\n{}\n{}\n{}\n{}\n{}\n{}", 
                neurotransmitters_definiton, 
                receptors.join("\n"),
                receptor_enum,
                receptors_struct,
                receptors_impl,
                receptors_default,
                default_impl,
            )
        )
    }

    #[cfg(feature = "gpu")]
    fn to_gpu_code(&self) -> (Vec<String>, String) {
        let neurotransmitter_gpu_impl = format!("
            impl NeurotransmitterTypeGPU for {}NeurotransmitterType {{
                fn type_to_numeric(&self) -> usize {{
                    match &self {{
                        {}
                    }}
                }}

                fn number_of_types() -> usize {{
                    {}
                }}

                fn get_all_types() -> BTreeSet<Self> {{
                    BTreeSet::from([
                        {}
                    ])
                }}

                fn to_string(&self) -> String {{
                    format!(\"{{:?}}\", self)
                }}
            }}

            impl {}NeurotransmitterType {{
                fn get_associated_receptor<T: ReceptorKinetics>(&self) -> {}Type<T> {{
                    match &self {{
                        {}
                    }}
                }}
            }}
            ",
            self.type_name.generate(),
            self.blocks.iter()
                .enumerate()
                .map(|(n, (current_type_name, _, _, _))| 
                    format!("{}NeurotransmitterType::{} => {}", self.type_name.generate(), current_type_name.generate(), n)
                )
                .collect::<Vec<_>>()
                .join(",\n"),
            self.blocks.len(),
            self.blocks.iter()
                .map(|(current_type_name, _, _, _)| 
                    format!("{}NeurotransmitterType::{}", self.type_name.generate(), current_type_name.generate())
                )
                .collect::<Vec<_>>()
                .join(",\n"),
            self.type_name.generate(),
            self.type_name.generate(),
            self.blocks.iter()
                .map(|(current_type_name, _, _, _)| 
                    format!(
                        "{}NeurotransmitterType::{} => {}Type::{}({}Receptor::<T>::default())", 
                        self.type_name.generate(), 
                        current_type_name.generate(), 
                        self.type_name.generate(), 
                        current_type_name.generate(), 
                        current_type_name.generate(), 
                    )
                )
                .collect::<Vec<_>>()
                .join(",\n")
        );

        let impl_header = format!("impl<T: ReceptorKineticsGPU> ReceptorsGPU for {}<T> {{", self.type_name.generate());

        let get_preprocessing_kinetics_parsing = "
            let split = attribute.split(\"$\").collect::<Vec<&str>>();
            if split.len() != 5 { return None; }
            let (receptor, neuro, name, kinetics, attr) = (split[0], split[1], split[2], split[3], split[4]);        
            if *receptor != *\"receptors\".to_string() || *kinetics != *\"kinetics\".to_string() { return None; }
            let stripped = format!(\"receptors$kinetics${}\", attr);
            let to_match = format!(\"{}${}\", neuro, name);
        ";
        let set_preprocessing_kinetics_parsing = "
            let split = attribute.split(\"$\").collect::<Vec<&str>>();
            if split.len() != 5 { return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, \"Invalid attribute\")); }
            let (receptor, neuro, name, kinetics, attr) = (split[0], split[1], split[2], split[3], split[4]);
            if *receptor != *\"receptors\".to_string() || *kinetics != *\"kinetics\".to_string() { return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, \"Invalid attribute\")); }
            let stripped = format!(\"receptors$kinetics${}\", attr);
            let to_match = format!(\"{}${}\", neuro, name);
        ";
      
        let mut neurotransmitters_to_receptor_vars: HashMap<String, Vec<String>> = HashMap::new();
        for (current_type, _, _, receptor_vars) in self.blocks.iter() {
            neurotransmitters_to_receptor_vars.insert(current_type.generate(), vec![]);
            if let Ast::VariablesAssignments(receptors) = receptor_vars {
                for name in receptors { 
                    neurotransmitters_to_receptor_vars.get_mut(&current_type.generate())
                        .unwrap()
                        .push(name.generate()); 
                }
            } else {
                unreachable!()
            } 
        }

        let get_kinetics = format!(
            "{{\n{}\nmatch to_match.as_str() {{\n{},\n_ => None}} }}",
            get_preprocessing_kinetics_parsing,
            generate_receptor_matching_inner_kinetics(
                &neurotransmitters_to_receptor_vars,
                &self.type_name.generate(),
            ).join(",\n"),
        );

        let set_kinetics = format!(
            "{{\n{}\nmatch to_match.as_str() {{\n{},\n_ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, \"Invalid attribute\"))}} }}",
            set_preprocessing_kinetics_parsing,
            generate_receptor_setting_inner_kinetics(
                &neurotransmitters_to_receptor_vars,
                &self.type_name.generate(),
            ).join(",\n"),
        );

        let get_attribute_header = "fn get_attribute(&self, attribute: &str) -> Option<BufferType> {";
        // check if type exists in current map, if it doesnt return none, else retrieve attribute
        let get_attribute_body = match &self.top_level_vars {
            Some(vars) => {
                format!(
                    "match attribute {{\n{},\n{},\n_ => {}\n}}",
                    generate_gpu_receptors_attribute_matching(vars, "top_").join(",\n"),
                    self.blocks.iter().map(|(current_type, current_vars, _, _)| {
                        generate_gpu_receptors_attribute_matching_inner_receptor(
                            current_vars, self.type_name.generate(), current_type.generate()
                        ).join(",\n")
                    })
                    .collect::<Vec<String>>()
                    .join(",\n"),
                    get_kinetics,
                )
            },
            None => {
                format!(
                    "match attribute {{\n{},\n_ => {}\n}}",
                    self.blocks.iter().map(|(current_type, current_vars, _, _)| {
                        generate_gpu_receptors_attribute_matching_inner_receptor(
                            current_vars, self.type_name.generate(), current_type.generate()
                        ).join(",\n")
                    })
                    .collect::<Vec<String>>()
                    .join(",\n"),
                    get_kinetics,
                )
            }
        };
        let get_attribute = format!("{}\n{}\n}}", get_attribute_header, get_attribute_body);

        let set_attribute_header = "fn set_attribute(&mut self, attribute: &str, value: BufferType) -> Result<(), std::io::Error> {";
        let set_attribute_body = match &self.top_level_vars {
            Some(vars) => {
                format!(
                    "match attribute {{\n{},\n{},\n_ => {}\n}};\nOk(())",
                    generate_gpu_receptors_attribute_setting(vars, "top_").join(",\n"),
                    self.blocks.iter().map(|(current_type, current_vars, _, _)| {
                        generate_gpu_receptors_attribute_setting_inner_receptor(
                            current_vars, self.type_name.generate(), current_type.generate()
                        ).join(",\n")
                    })
                    .collect::<Vec<String>>()
                    .join(",\n"),
                    set_kinetics,
                )
            },
            None => {
                format!(
                    "match attribute {{\n{},\n_ => {}\n}};\nOk(())",
                    self.blocks.iter().map(|(current_type, current_vars, _, _)| {
                        generate_gpu_receptors_attribute_setting_inner_receptor(
                            current_vars, self.type_name.generate(), current_type.generate()
                        ).join(",\n")
                    })
                    .collect::<Vec<String>>()
                    .join(",\n"),
                    set_kinetics,
                )
            }
        };
        let set_attribute = format!("{}\n{}\n}}", set_attribute_header, set_attribute_body);

        // generate all attribute names
        // use T to list all attributes in each receptor var, and parse
        // out the individual attrs and add them to the hashset

        let get_all_attributes_header = "fn get_all_attributes() -> HashSet<(String, AvailableBufferType)> {";
        // for i in top level vars; for i in block { for var in block; for receptor in block { for attr in receptor } }

        let mut all_attrs = match &self.top_level_vars {
            Some(ref vals) => generate_gpu_receptors_attributes_vec(vals, "top_"),
            None => vec![]
        };

        let mut all_receptor_attrs_generation = vec![];

        let top_level_attrs = all_attrs.clone();

        let mut neurotransmitter_to_get_attr: HashMap<String, (Vec<String>, Vec<String>)> = HashMap::new();

        for (current_type, current_vars, _, receptor_vars) in self.blocks.iter() {
            let neuro_prefix = format!("{}_", current_type.generate());
            let current_attrs = generate_gpu_receptors_attributes_vec(current_vars, &neuro_prefix);

            all_attrs.extend(current_attrs.clone());

            let mut receptor_attrs_generation = vec![];

            if let Ast::VariablesAssignments(receptor_var_names) = receptor_vars {
                for name in receptor_var_names {
                    let receptor_kinetics_vars = format!(
                        "attrs.extend(
                            T::get_attribute_names().iter().map(|(i, j)|
                                (
                                    format!(
                                        \"receptors${}${}$kinetics${{}}\", 
                                        i.split(\"$\").collect::<Vec<_>>().last()
                                            .expect(\"Invalid attribute\")
                                    ), 
                                    *j
                                )
                            ).collect::<Vec<(_, _)>>()
                        );", 
                        current_type.generate(), 
                        &name.generate()[5..], 
                    );

                    receptor_attrs_generation.push(receptor_kinetics_vars.clone());
                    all_receptor_attrs_generation.push(receptor_kinetics_vars);
                }
            } else {
                unreachable!()
            }

            neurotransmitter_to_get_attr.insert(
                format!(
                    "{}NeurotransmitterType::{}", 
                    self.type_name.generate(),
                    current_type.generate(),
                ),
                (
                    current_attrs.clone(),
                    receptor_attrs_generation,
                )
            );
        }

        let get_all_attributes = format!(
            "{}\nlet mut attrs = HashSet::from([{}]);\n{}\nattrs\n}}", 
            get_all_attributes_header,
            all_attrs.join(", "),
            all_receptor_attrs_generation.join("\n"),
        );

        // convert to gpu
        // convert to cpu
        // needs to generate and read receptor flag vars

        let convert_to_gpu = "fn convert_to_gpu(
            grid: &[Vec<Self>], context: &Context, queue: &CommandQueue
        ) -> Result<HashMap<String, BufferGPU>, GPUError> {
            if grid.is_empty() || grid.iter().all(|i| i.is_empty()) {
                return Ok(HashMap::new());
            }

            let mut buffers = HashMap::new();

            let size: usize = grid.iter().map(|row| row.len()).sum();
            
            for (attr, current_type) in Self::get_all_attributes() {
                match current_type {
                    AvailableBufferType::Float => {
                        let mut current_attrs: Vec<f32> = vec![];
                        for row in grid.iter() {
                            for i in row.iter() {
                                match i.get_attribute(&attr) {
                                    Some(BufferType::Float(val)) => current_attrs.push(val),
                                    Some(_) => unreachable!(),
                                    None => current_attrs.push(0.),
                                };
                            }
                        }

                        write_buffer!(current_buffer, context, queue, size, &current_attrs, Float, last);

                        buffers.insert(attr.clone(), BufferGPU::Float(current_buffer));
                    },
                    AvailableBufferType::UInt => {
                        let mut current_attrs: Vec<u32> = vec![];
                        for row in grid.iter() {
                            for i in row.iter() {
                                match i.get_attribute(&attr) {
                                    Some(BufferType::UInt(val)) => current_attrs.push(val),
                                    Some(_) => unreachable!(),
                                    None => current_attrs.push(0),
                                };
                            }
                        }

                        write_buffer!(current_buffer, context, queue, size, &current_attrs, UInt, last);

                        buffers.insert(attr.clone(), BufferGPU::UInt(current_buffer));
                    },
                    _ => unreachable!(),
                }
            }

            let mut receptor_flags: Vec<u32> = vec![];
            for row in grid.iter() {
                for i in row.iter() {
                    for n in <Self as Receptors>::N::get_all_types() {
                        match i.receptors.get(&n) {
                            Some(_) => receptor_flags.push(1),
                            None => receptor_flags.push(0),
                        };
                    }
                }
            }

            let flags_size = size * <Self as Receptors>::N::number_of_types();

            write_buffer!(flag_buffer, context, queue, flags_size, &receptor_flags, UInt, last);

            buffers.insert(String::from(\"receptors$flags\"), BufferGPU::UInt(flag_buffer));

            Ok(buffers)
        }";

        let get_all_top_level_attributes = format!(
            "fn get_all_top_level_attributes() -> HashSet<(String, AvailableBufferType)> {{
                HashSet::from([{}])
            }}",
            top_level_attrs.join(", "),
        );

        // match on neurotransmitter type
        // return associated vars and associated receptor kinetics vars with that neurotransmitter type
        let get_attributes_associated_with = format!(
            "fn get_attributes_associated_with(neurotransmitter: &{}NeurotransmitterType) -> HashSet<(String, AvailableBufferType)> {{
                match neurotransmitter {{
                    {}
                }}
            }}",
            self.type_name.generate(),
            neurotransmitter_to_get_attr.iter().map(|(i, (attrs, receptor_vars))| {
                format!(
                    "{} => {{\nlet mut attrs = HashSet::from([{}]);\n{}\nattrs\n}}",
                    i,
                    attrs.join(", "),
                    receptor_vars.join("\n"),
                )
            }).collect::<Vec<_>>().join(",\n")
        );

        let convert_to_cpu = "
            fn convert_to_cpu(
                grid: &mut [Vec<Self>],
                buffers: &HashMap<String, BufferGPU>,
                queue: &CommandQueue,
                rows: usize,
                cols: usize,
            ) -> Result<(), GPUError> {
                if rows == 0 || cols == 0 {
                    for inner in grid {
                        inner.clear();
                    }

                    return Ok(());
                }

                let mut cpu_conversion: HashMap<String, Vec<BufferType>> = HashMap::new();

                for key in Self::get_all_attributes() {
                    match key.1 {
                        AvailableBufferType::Float => {
                            let mut current_contents = vec![0.; rows * cols];
                            read_and_set_buffer!(buffers, queue, &key.0, &mut current_contents, Float);

                            let current_contents = current_contents.iter()
                                .map(|i| BufferType::Float(*i))
                                .collect::<Vec<BufferType>>();

                            cpu_conversion.insert(key.0.clone(), current_contents);
                        },
                        AvailableBufferType::UInt => {
                            let mut current_contents = vec![0; rows * cols];
                            read_and_set_buffer!(buffers, queue, &key.0, &mut current_contents, UInt);

                            let current_contents = current_contents.iter()
                                .map(|i| BufferType::UInt(*i))
                                .collect::<Vec<BufferType>>();

                            cpu_conversion.insert(key.0.clone(), current_contents);
                        },
                        _ => unreachable!(),
                    }
                }

                let mut current_contents = vec![0; rows * cols * <Self as Receptors>::N::number_of_types()];
                read_and_set_buffer!(buffers, queue, \"receptors$flags\", &mut current_contents, UInt);

                let flags = current_contents.iter().map(|i| *i == 1).collect::<Vec<bool>>();

                for row in 0..rows {
                    for col in 0..cols {
                        let current_index = row * cols + col;

                        for i in Self::get_all_top_level_attributes() {
                            grid[row][col].set_attribute(
                                &i.0, 
                                cpu_conversion.get(&i.0).unwrap()[current_index]
                            ).unwrap();
                        }
                        for i in <Self as Receptors>::N::get_all_types() {
                            if flags[current_index * <Self as Receptors>::N::number_of_types() + i.type_to_numeric()] {
                                for attr in Self::get_attributes_associated_with(&i) {
                                    match grid[row][col].receptors.get_mut(&i) {
                                        Some(_) => grid[row][col].set_attribute(
                                            &attr.0, 
                                            cpu_conversion.get(&attr.0).unwrap()[current_index]
                                        ).unwrap(),
                                        None => {
                                            grid[row][col].receptors.insert(
                                                i,
                                                i.get_associated_receptor()
                                            );
                                            grid[row][col].set_attribute(
                                                &attr.0, 
                                                cpu_conversion.get(&attr.0).unwrap()[current_index]
                                            ).unwrap();
                                        }
                                    };
                                }
                            } else {
                                let _ = grid[row][col].receptors.remove(&i);
                            }
                        }
                    }
                }

                Ok(())
            }
        ";

        // fns for each receptor iteration type
        // when generating neuron type
        // iterate over each function to add to main program
        // iterate over each function signature to add to receptor updates
        // need to also update the receptor kinetics depending on input

        // need a ReceptorsGPU trait to associate neurotransmitter N type

        // function itself, arg identifiers, arg type
        let mut update_blocks: Vec<(String, Vec<String>)> = vec![];

        for (current_type, current_vars, on_iteration, receptor_vars) in self.blocks.iter() {
            let mut current_receptor_vars = vec![];
            if let Ast::VariablesAssignments(vars) = receptor_vars {
                for i in vars {
                    current_receptor_vars.push(i);
                }
            }

            let signature = if let Some(top_level) = &self.top_level_vars {
                format!(
                    "__kernel void update_{}(uint index, __global float *current_voltage, __global float *dt, {}, {}, {}) {{",
                    current_type.generate(),
                    generate_kernel_args(top_level).join(", "),
                    current_receptor_vars.iter().map(|i| format!("__global float *{}", &i.generate()[5..]))
                        .collect::<Vec<_>>().join(", "),
                    generate_kernel_args(current_vars).join(", "),
                )
            } else {
                format!(
                    "__kernel void update_{}(uint index, __global float *current_voltage, __global float *dt, {}, {}) {{",
                    current_type.generate(),
                    current_receptor_vars.iter().map(|i| format!("__global float *{}", &i.generate()[5..]))
                        .collect::<Vec<_>>().join(", "),
                    generate_kernel_args(current_vars).join(", "),
                )
            };

            let body = generate_gpu_kernel_on_iteration(on_iteration);

            let function = format!("{}\n{}\n}}", signature, body);

            let mut args = vec![
                String::from("(String::from(\"index\"), AvailableBufferType::UInt)"),
                String::from("(String::from(\"current_voltage\"), AvailableBufferType::Float)"),
                String::from("(String::from(\"dt\"), AvailableBufferType::Float)"),
            ];
            if let Some(top_level) = &self.top_level_vars {
                args.extend(
                    generate_gpu_receptors_attributes_vec(
                        top_level, "top_"
                    )
                );
            }
            args.extend(
                current_receptor_vars.iter()
                    .map(|i| format!(
                        "(String::from(\"receptors${}${}$kinetics$r\"), AvailableBufferType::Float)", 
                        current_type.generate(),
                        &i.generate()[5..],
                    ))
                    .collect::<Vec<String>>()
            );
            args.extend(
                generate_gpu_receptors_attributes_vec(
                    current_vars, &format!("{}_", current_type.generate())
                )
            );

            update_blocks.push((function, args));
        }

        let get_updates = format!(
            "fn get_updates() -> Vec<(String, Vec<(String, AvailableBufferType)>)> {{
                vec![{}]
            }}",
            update_blocks.iter().map(|(i, j)| {
                format!(
                    "(String::from(\"{}\"), vec![{}])",
                    i,
                    j.join(", ")
                )
            }).collect::<Vec<String>>().join(", ")
        );

        let imports = vec![
            String::from("use std::collections::HashSet;"),
            String::from("use std::collections::BTreeSet;"),
            String::from("use std::ptr;"),
            String::from("use opencl3::command_queue::CommandQueue;"),
            String::from("use opencl3::context::Context;"),
            String::from("use opencl3::memory::Buffer;"),
            String::from("use opencl3::memory::CL_MEM_READ_WRITE;"),
            String::from("use opencl3::types::cl_uint;"),
            String::from("use opencl3::types::cl_float;"),
            String::from("use opencl3::types::CL_BLOCKING;"),
            String::from("use opencl3::types::CL_NON_BLOCKING;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::NeurotransmitterTypeGPU;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::ReceptorKineticsGPU;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::ReceptorsGPU;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::BufferType;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::AvailableBufferType;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::write_buffer;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::read_and_set_buffer;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::BufferGPU;"),
            String::from("use spiking_neural_networks::neuron::iterate_and_spike::BufferType;"),
            String::from("use spiking_neural_networks::error::GPUError;"),
        ];

        // function that returns update blocks with args for neuron to use

        (
            imports,
            format!(
                "
                {}
                {}
                {}
                {}
                {}
                {}
                {}
                {}
                {}
                {}
                }}
                ",
                neurotransmitter_gpu_impl,
                impl_header,
                get_attribute,
                set_attribute,
                get_all_attributes,
                convert_to_gpu,
                get_all_top_level_attributes,
                get_attributes_associated_with,
                convert_to_cpu,
                get_updates,
            )
        )
    }

    #[cfg(feature = "py")]
    fn to_pyo3_code(&self) -> (Vec<String>, String) {
        let neurotransmitters_name = format!("{}NeurotransmitterType", self.type_name.generate());

        let neurotransmitter_types: Vec<String> = self.blocks.iter()
            .map(|i| i.0.generate())
            .collect();
        let neurotransmitters_struct_def = format!(
            "#[pyclass]
            #[pyo3(name = \"{}\")]
            #[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]\npub enum Py{} {{\n{}\n}}",
            neurotransmitters_name,
            neurotransmitters_name,
            neurotransmitter_types.join(",\n"),
        );

        let neurotransmitter_conversions = neurotransmitter_types.iter()
            .map(|i| format!(
                "Py{}::{} => {}::{}", 
                neurotransmitters_name,
                i,
                neurotransmitters_name,
                i,
            ))
            .collect::<Vec<_>>();

        let neurotransmitter_conversions_reverse = neurotransmitter_types.iter()
            .map(|i| format!(
                "{}::{} => Py{}::{}", 
                neurotransmitters_name,
                i,
                neurotransmitters_name,
                i,
            ))
            .collect::<Vec<_>>();

        let neurotransmitter_conversion_impl = format!(
            "impl Py{} {{
                pub fn convert_type(&self) -> {} {{
                    match self {{
                        {}
                    }}
                }}
            }}",
            neurotransmitters_name,
            neurotransmitters_name,
            neurotransmitter_conversions.join(",\n"),
        );

        let neurotransmitter_conversion_reverse_impl = format!(
            "impl {} {{
                pub fn convert_type_to_py(&self) -> Py{} {{
                    match self {{
                        {}
                    }}
                }}

                pub fn convert_from_py(neurotransmitter: &PyAny) -> Option<Self> {{
                    neurotransmitter.extract::<Py{}>().ok()
                        .map(|i| i.convert_type())
                }}
            }}",
            neurotransmitters_name,
            neurotransmitters_name,
            neurotransmitter_conversions_reverse.join(",\n"),
            neurotransmitters_name,
        );

        let neurotransmitter_py_impl = format!(
            "#[pymethods]
            impl Py{} {{
                fn __hash__(&self) -> u64 {{
                    let mut hasher = DefaultHasher::new();
                    self.hash(&mut hasher);
                    hasher.finish()
                }}
            }}",
            neurotransmitters_name,
        );

        let default_kinetics = self.default_kinetics.as_ref()
            .unwrap_or(&Ast::SingleKineticsDefinition(String::from("ApproximateReceptor")))
            .generate();

        let mut receptor_impls = vec![];
        let mut receptor_conversions = vec![];
        let mut receptor_reverse_conversions = vec![];
        let mut has_current = false;

        for (type_name, vars_def, _, receptor_vars) in &self.blocks {
            let vars = generate_fields(vars_def);
            if vars.contains(&String::from("pub current: f32")) {
                has_current = true;
            }

            let struct_def = format!(
                "#[pyclass]
                #[pyo3(name = \"{}Receptor\")]
                #[derive(Debug, Clone, PartialEq)]\npub struct Py{}Receptor {{\nreceptor: {}Receptor<{}>\n}}", 
                type_name.generate(),
                type_name.generate(),
                type_name.generate(),
                default_kinetics,
            );

            let getters_and_setters = generate_vars_as_getter_setters("receptor", vars_def);
            let receptor_getters_and_setters = generate_receptor_vars_as_getter_setters(
                format!("Py{}", default_kinetics).as_str(), receptor_vars
            );

            let iterate_function = match &self.top_level_vars {
                Some(top_level_vars) => format!(
                    "fn iterate(&mut self, current_voltage: f32, dt: f32, {}) {{ {}\nself.receptor.iterate(current_voltage, dt, {}); }}",
                    generate_fields_as_immutable_args(top_level_vars).join(", "),
                    generate_fields_as_mutable_statements("processed", top_level_vars).join("\n"),
                    generate_fields_as_mutable_refs("processed", top_level_vars).join(", "),
                ),
                None => String::from(
                    "fn iterate(&mut self, current_voltage: f32, dt: f32) {{ self.receptor.iterate(current_voltage, dt); }}"
                ),
            };

            let py_impl = format!(
                "#[pymethods]
                impl Py{}Receptor {{
                    #[new]
                    #[pyo3(signature = ({}, {}))]
                    fn new({}, {}) -> Self {{ 
                        Py{}Receptor {{ 
                            receptor: {}Receptor {{
                                {},
                                {},
                            }}
                        }} 
                    }}
                    fn __repr__(&self) -> PyResult<String> {{ Ok(format!(\"{{:#?}}\", self.receptor)) }}
                    {}
                    {}
                    fn apply_r_changes(&mut self, t: f32, dt: f32) {{ self.receptor.apply_r_change(t, dt); }}
                    {}
                }}
                ",
                type_name.generate(),
                generate_fields_as_fn_new_args(vars_def).join(", "),
                generate_py_receptors_as_fn_new_args(&default_kinetics, receptor_vars).join(","),
                generate_fields_as_immutable_args(vars_def).join(", "),
                generate_py_receptors_as_args(&format!("Py{}", default_kinetics), receptor_vars).join(","),
                type_name.generate(),
                type_name.generate(),
                generate_fields_as_names(vars_def).join(",\n"),
                generate_py_receptors_as_args_in_receptor(receptor_vars).join(","),
                getters_and_setters.join("\n"),
                receptor_getters_and_setters.join("\n"),
                iterate_function,
            );

            receptor_impls.push(struct_def);
            receptor_impls.push(py_impl);

            receptor_conversions.push(
                format!(
                    "Some({}Type::{}(current_receptor)) => 
                    Py::new(py, Py{}Receptor {{ receptor: current_receptor.clone() }}).unwrap().into_py(py)", 
                    self.type_name.generate(),
                    type_name.generate(),
                    type_name.generate(),
                )
            );
            receptor_reverse_conversions.push(
                format!(
                    "match receptor.extract::<Py{}Receptor>() {{
                        Ok(val) => {{
                            match self.receptors.insert(current_type, {}Type::{}(val.receptor.clone())) {{
                                Ok(_) => return Ok(()),
                                Err(_) => return Err(PyTypeError::new_err(\"Incorrect neurotransmitter type\")),
                            }}
                        }},
                        Err(e) => {{}},
                    }};",
                    type_name.generate(),
                    self.type_name.generate(),
                    type_name.generate(),
                )
            );
        }

        let receptor_struct_def = format!(
            "#[pyclass]
            #[pyo3(name = \"{}\")]
            #[derive(Debug, Clone)]
            pub struct Py{} {{
                receptors: {}<{}>
            }}",
            self.type_name.generate(),
            self.type_name.generate(),
            self.type_name.generate(),
            default_kinetics,
        );

        let top_level_getter_setters = match &self.top_level_vars {
            Some(vars) => generate_vars_as_getter_setters("receptors", vars).join("\n"),
            None => String::from(""),
        };

        let update_receptor_kinetics = format!(
            "fn update_receptor_kinetics(&mut self, t: &PyDict, dt: f32) -> PyResult<()> {{
                let mut conc = HashMap::new();
                for (key, value) in t.iter() {{
                    let current_type = {}::convert_from_py(key);
                    if current_type.is_none() {{
                        return Err(PyTypeError::new_err(\"Incorrect neurotransmitter type\"));
                    }}
                    let current_t = value.extract::<f32>()?;
                    conc.insert(
                        current_type.unwrap(), 
                        current_t
                    );
                }}
        
                self.receptors.update_receptor_kinetics(&conc, dt);
        
                Ok(())
            }}",
            neurotransmitters_name,
        );

        let get_receptor_currents = if has_current {
            "fn get_receptor_currents(&self, dt: f32, c_m: f32) -> f32 {
                self.receptors.get_receptor_currents(dt, c_m)
            }"
        } else {
            ""
        };

        let receptor_py_impl = format!(
            "#[pymethods]
            impl Py{} {{
                #[new]
                fn new() -> Self {{ Py{} {{ receptors: {}::default_impl() }} }}
                {}
                fn __repr__(&self) -> PyResult<String> {{ Ok(format!(\"{{:#?}}\", self.receptors)) }}
                fn __len__(&self) -> usize {{ self.receptors.len() }}
                {}
                fn set_receptor_currents(&mut self, current_voltage: f32, dt: f32) {{
                    self.receptors.set_receptor_currents(current_voltage, dt);
                }}
                {}
                fn remove(&mut self, neurotransmitter_type: Py{}) {{
                    self.receptors.remove(&neurotransmitter_type.convert_type()).unwrap();
                }}
                fn get<'py>(&self, py: Python<'py>, neurotransmitter_type: Py{}) -> Py<PyAny> {{
                    let receptor = self.receptors.get(&neurotransmitter_type.convert_type());
                    match receptor {{
                        {},
                        None => py.None(),
                    }}
                }}
                fn insert(&mut self, neurotransmitter_type: Py{}, receptor: &PyAny) -> PyResult<()> {{
                    let current_type = neurotransmitter_type.convert_type();

                    {}

                    Err(PyTypeError::new_err(\"Receptor type is unknown\"))
                }}
            }}",
            self.type_name.generate(),
            self.type_name.generate(),
            self.type_name.generate(),
            top_level_getter_setters,
            update_receptor_kinetics,
            get_receptor_currents,
            neurotransmitters_name,
            neurotransmitters_name,
            receptor_conversions.join(",\n"),
            neurotransmitters_name,
            receptor_reverse_conversions.join("\n"),
        );
    
        let imports = vec![
            String::from("use pyo3::prelude::*;"),
            String::from("use pyo3::types::PyDict;"),
            String::from("use pyo3::exceptions::PyTypeError;"),
            String::from("use std::collections::hash_map::DefaultHasher;"),
            String::from("use std::hash::Hash;"),
            String::from("use std::hash::Hasher;"),
        ];

        (
            imports,
            format!(
                "{}
                {}
                {}
                {}
                {}
                {}
                {}",
                neurotransmitters_struct_def,
                neurotransmitter_conversion_impl,
                neurotransmitter_conversion_reverse_impl,
                neurotransmitter_py_impl,
                receptor_impls.join("\n"),
                receptor_struct_def,
                receptor_py_impl,
            )
        )
    }
}

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
            Rule::bool => Ast::Bool(primary.as_str().parse::<bool>().unwrap()),
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
                .expect("Could not get what to assign to").as_str()
            );

            let eq_operator: String = String::from(inner_rules.next()
                .expect("Could not get equation operator name").as_str()
            );

            let expr: Box<Ast> = Box::new(
                parse_bool_expr(
                    inner_rules.next()
                        .expect("No arguments found")
                        .into_inner()
                )
            );

            Ast::EqAssignment { name, eq_operator, expr }
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

            let mut conditions = vec![parse_bool_expr(inner_rules.next().unwrap().into_inner())];

            let mut declarations: Vec<Vec<Ast>> = vec![
                inner_rules
                    .next()
                    .unwrap()
                    .into_inner()
                    .map(|i| parse_declaration(i))
                    .collect::<Vec<Ast>>()
            ];

            for inner in inner_rules {
                match inner.as_rule() {
                    Rule::else_if_body => {
                        let mut inner_pairs = inner.into_inner();

                        let condition = parse_bool_expr(inner_pairs.next().unwrap().into_inner());

                        let decls = inner_pairs
                            .map(|i| parse_declaration(i))
                            .collect::<Vec<Ast>>();

                        conditions.push(condition);
                        declarations.push(decls);
                    }
                    Rule::else_body => {
                        declarations.push(
                            inner
                                .into_inner()
                                .map(|i| parse_declaration(i))
                                .collect::<Vec<Ast>>()
                        );
                    }
                    _ => unreachable!("Non if statement found")
                }
            }

            Ast::IfStatement { conditions, declarations }
        }
        rule => unreachable!("Unexpected declaration, found {:#?}", rule),
    }
}

// fn extract_name_from_pattern(string: &str, i: &str) -> Vec<String> {
//     let re = Regex::new(&format!(r"pub (.*): {}", i)).unwrap();
//     let mut output = vec![];

//     for caps in re.captures_iter(string) {
//         let first_part = &caps[1];
//         if string.contains(i) {
//             output.push(first_part.to_string());
//         }
//     }

//     output
// }

// fn insert_at_substring(original: &str, to_find: &str, to_insert: &str) -> String {
//     if let Some(start) = original.find(to_find) {
//         let mut result = String::new();
//         result.push_str(&original[..start + to_find.len()]);
//         result.push_str(to_insert);
//         result.push_str(&original[start + to_find.len()..]);

//         result
//     } else {
//         String::from(original)
//     }
// }

fn build_function(model_description: String) -> TokenStream {
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
    
                        for i in &neuron_imports {
                            if !imports.contains(i) {
                                imports.push(i.clone());
                            }
                        }
    
                        let neuron_type_name = neuron_definition.type_name.generate();
    
                        let neuron_code_map = code.entry(String::from("neuron"))
                            .or_default();
                    
                        neuron_code_map.insert(neuron_type_name, neuron_code);

                        #[cfg(feature = "gpu")]
                        {
                            let (neuron_imports, neuron_code) = neuron_definition.to_gpu_code();

                            for i in neuron_imports {
                                if !imports.contains(&i) {
                                    imports.push(i);
                                }
                            }
    
                            neuron_code_map.insert(
                                format!("{}GPU", neuron_definition.type_name.generate()), neuron_code
                            );
                        }

                        #[cfg(feature = "py")]
                        {
                            let (neuron_imports, neuron_code) = neuron_definition.to_pyo3_code();

                            for i in neuron_imports {
                                if !imports.contains(&i) {
                                    imports.push(i);
                                }
                            }
    
                            neuron_code_map.insert(
                                format!("{}PY", neuron_definition.type_name.generate()), neuron_code
                            );
                        }
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

                        #[cfg(feature = "gpu")]
                        {
                            let (ion_channel_imports, ion_channel_code) = ion_channel.to_gpu_code();

                            for i in ion_channel_imports {
                                if !imports.contains(&i) {
                                    imports.push(i);
                                }
                            }
    
                            ion_channel_code_map.insert(
                                format!("{}GPU", ion_channel.type_name.generate()), ion_channel_code
                            );
                        }

                        #[cfg(feature = "py")]
                        {
                            let (ion_channel_imports, ion_channel_code) = ion_channel.to_pyo3_code();

                            for i in ion_channel_imports {
                                if !imports.contains(&i) {
                                    imports.push(i);
                                }
                            }
    
                            ion_channel_code_map.insert(
                                format!("{}PY", ion_channel.type_name.generate()), ion_channel_code
                            );
                        }
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

                        #[cfg(feature="gpu")] 
                        {
                            let (neurotransmitter_kinetics_imports, neurotransmitter_kinetics_code) = neurotransmitter_kinetics.to_gpu_code();

                            for i in neurotransmitter_kinetics_imports {
                                if !imports.contains(&i) {
                                    imports.push(i);
                                }
                            }
    
                            neurotransmitter_kinetics_code_map.insert(
                                format!("{}GPU", neurotransmitter_kinetics.type_name.generate()), neurotransmitter_kinetics_code
                            );
                        }

                        #[cfg(feature = "py")]
                        {
                            let (neurotransmitter_kinetics_imports, neurotransmitter_kinetics_code) = neurotransmitter_kinetics.to_pyo3_code();

                            for i in neurotransmitter_kinetics_imports {
                                if !imports.contains(&i) {
                                    imports.push(i);
                                }
                            }
    
                            neurotransmitter_kinetics_code_map.insert(
                                format!("{}PY", neurotransmitter_kinetics.type_name.generate()), neurotransmitter_kinetics_code
                            );
                        }
                    },
                    Rule::receptor_kinetics_definition => {
                        let receptor_kinetics = generate_receptor_kinetics(pair.into_inner())
                            .expect("Could not generate receptor kinetics");

                        let (receptor_kinetics_imports, receptor_kinetics_code) = receptor_kinetics.to_code();

                        for i in receptor_kinetics_imports {
                            if !imports.contains(&i) {
                                imports.push(i);
                            }
                        }
    
                        let receptor_kinetics_type_name = receptor_kinetics.type_name.generate();
                    
                        let receptor_kinetics_code_map = code.entry(String::from("receptor_kinetics"))
                            .or_default();
    
                        receptor_kinetics_code_map.insert(
                            receptor_kinetics_type_name, receptor_kinetics_code
                        );

                        #[cfg(feature="gpu")] 
                        {
                            let (receptor_kinetics_imports, receptor_kinetics_code) = receptor_kinetics.to_gpu_code();

                            for i in receptor_kinetics_imports {
                                if !imports.contains(&i) {
                                    imports.push(i);
                                }
                            }
    
                            receptor_kinetics_code_map.insert(
                                format!("{}GPU", receptor_kinetics.type_name.generate()), receptor_kinetics_code
                            );
                        }

                        #[cfg(feature="py")] 
                        {
                            let (receptor_kinetics_imports, receptor_kinetics_code) = receptor_kinetics.to_pyo3_code();

                            for i in receptor_kinetics_imports {
                                if !imports.contains(&i) {
                                    imports.push(i);
                                }
                            }
    
                            receptor_kinetics_code_map.insert(
                                format!("{}PY", receptor_kinetics.type_name.generate()), receptor_kinetics_code
                            );
                        }
                    },
                    Rule::receptors_definition => {
                        let receptors = generate_receptors(pair.into_inner())
                            .expect("Could not generate receptor kinetics");

                        let (receptors_imports, receptors_code) = receptors.to_code();

                        for i in receptors_imports {
                            if !imports.contains(&i) {
                                imports.push(i);
                            }
                        }
    
                        let receptors_type_name = receptors.type_name.generate();
                    
                        let receptors_code_map = code.entry(String::from("receptors"))
                            .or_default();
    
                        receptors_code_map.insert(
                            receptors_type_name, receptors_code
                        );

                        #[cfg(feature="gpu")] 
                        {
                            let (receptors_imports, receptors_code) = receptors.to_gpu_code();

                            for i in receptors_imports {
                                if !imports.contains(&i) {
                                    imports.push(i);
                                }
                            }
    
                            receptors_code_map.insert(
                                format!("{}GPU", receptors.type_name.generate()), receptors_code
                            );
                        }

                        #[cfg(feature="py")] 
                        {
                            let (receptors_imports, receptors_code) = receptors.to_pyo3_code();

                            for i in receptors_imports {
                                if !imports.contains(&i) {
                                    imports.push(i);
                                }
                            }
    
                            receptors_code_map.insert(
                                format!("{}PY", receptors.type_name.generate()), receptors_code
                            );
                        }
                    }
                    Rule::EOI => {
                        continue
                    }
                    _ => unreachable!("Unexpected definition: {:#?}", pair.as_rule()),
                }
            }
        
            // if any of the ion channel names found in neuron
            // (use substring to detect)
            // modify neuron code to insert proper update current code before dv changes
    
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

            // eprintln!("{}\n\n\n{}", imports.join("\n"), all_code);
        
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

fn parse_out_comments(string: &str) -> String {
    if string.len() < 2 {
        return string.to_string();
    }

    let mut output = String::new();

    let mut is_comment = false;
    let mut is_comment_inline = false;
    for i in 0..string.len() {
        if string.chars().nth(i).unwrap() == '/' && string.chars().nth(i + 1) == Some('/') {
            is_comment = true;
        } else if string.chars().nth(i).unwrap() == '/' && string.chars().nth(i + 1) == Some('*') {
            is_comment_inline = true;
        }
        if !is_comment && !is_comment_inline {
            output.push(string.chars().nth(i).unwrap());
        } else if is_comment && string.chars().nth(i).unwrap() == '\n' {
            is_comment = false;
            output.push('\n');
        } else if is_comment_inline && string.chars().nth(i).unwrap() == '/' && 
            string.chars().nth(i - 1).unwrap() == '*' {
            is_comment_inline = false;
        }
    }

    output
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

    build_function(parse_out_comments(model_description.as_str()))
}

#[proc_macro]
pub fn neuron_builder_from_file(filename: TokenStream) -> TokenStream {
    let filename = parse_macro_input!(filename as LitStr);
    let filename = filename.value();

    build_function(parse_out_comments(
        read_to_string(filename).expect("Could not read file to string").as_str()
    ))
}

#[cfg(test)]
mod test {
    use crate::parse_out_comments;

    
    #[test]
    fn test_comment_strip() {
        let string = "hi hello\nhow are you // comment\nend of comment hopefully";
        let expected = "hi hello\nhow are you \nend of comment hopefully";

        assert_eq!(expected, parse_out_comments(string));
    }

    #[test]
    fn test_comment_strip_no_comments() {
        let string = "aoisdf/n 1\n8w23fna948\nasdl$fjaslk/df3hq238f*732fonwauds  aksdjfhaw\nap38 w2";
        
        assert_eq!(string, parse_out_comments(string));
    }

    #[test]
    fn test_comment_strip_many_comments() {
        let string = "[neuron]\ntype: n // test\nkinetics: default\nvars: another = 1 // // stuff\n[end] // extra";
        let expected = "[neuron]\ntype: n \nkinetics: default\nvars: another = 1 \n[end] ";

        assert_eq!(expected, parse_out_comments(string));
    }

    #[test]
    fn test_strip_comment_end_of_string() {
        let string = "asdfjasldjf stuff // comment";
        let expected = "asdfjasldjf stuff ";

        assert_eq!(expected, parse_out_comments(string));
    }

    #[test]
    fn test_strip_inline_comment() {
        let string = "aslkdfjap892rioavnalsjdf;l/* stuff */adfasjl283yref9q83phneavasfhsjakdfk";
        let expected = "aslkdfjap892rioavnalsjdf;ladfasjl283yref9q83phneavasfhsjakdfk";

        assert_eq!(expected, parse_out_comments(string));
    }

    #[test]
    fn test_comment_strip_many_inline() {
        let string = "[neuron]\ntype: /* neuron */ n\nvars: g = 1 /* g_max */, e = 1\non_spike: is_spiking /* stuff */\n[end] /* other */";
        let expected = "[neuron]\ntype:  n\nvars: g = 1 , e = 1\non_spike: is_spiking \n[end] ";

        assert_eq!(expected, parse_out_comments(string));
    }

    #[test]
    fn test_comment_strip_mixed() {
        let string = "hello // comment\nyadadada /* inline */ followed by // comment\nnextline // /* nested */ yea\n";
        let expected = "hello \nyadadada  followed by \nnextline \n";

        assert_eq!(expected, parse_out_comments(string));
    }
}
