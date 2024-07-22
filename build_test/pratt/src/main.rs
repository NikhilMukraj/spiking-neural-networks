use std::collections::HashMap;
use std::io::{Error, ErrorKind};
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
    Variables(Vec<Box<AST>>),
}

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
            AST::Name(name) => name.clone(),
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
                format!("{} = {}", name, expr.to_string())
            },
            AST::DiffEqAssignment { name, expr } => {
                format!("d{}/dt = {}", name, expr.to_string())
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
            AST::TypeDefinition(string) => format!("type: {}", string),
            AST::OnSpike(assignments) => {
                let assignments_string = assignments.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
                    .join("\n");

                format!("on_spike:\n{}", assignments_string)
            },
            AST::OnIteration(assignments) => {
                let assignments_string = assignments.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
                    .join("\n");

                format!("on_iteration:\n{}", assignments_string)
            },
            AST::SpikeDetection(expr) => {
                format!("spike_detection: {}", expr.to_string())
            },
            AST::Variables(assignments) => {
                let assignments_string = assignments.iter()
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join("\n");

                format!("vars:\n{}", assignments_string)
            }
        }
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
    // runge kutta

    // handle ion channels
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
                            AST::OnIteration(
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
                    // Rule::vars_def => {
                        // if no defaults then just assume assingment is None
                        // in order to prevent duplicate, key should be "vars"
                    // }
                    // Rule::vars_with_default_def => {
                        // assignment should be just a number
                        // in order to prevent duplicate, key should be "vars"
                    // }
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

            // neuron definition as part of ast enum
            // anything that has a default can be represented as an option
            // if none use default version of field

            for value in definitions.values() {
                println!("{}", value.to_string());
            }
        }
        Err(e) => {
            eprintln!("Parse failed: {:?}", e);
        }
    }

    Ok(())
}
