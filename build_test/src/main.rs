use std::env;
use pest::iterators::Pairs;
use pest::pratt_parser::PrattParser;
use pest::Parser;
use std::io::{self, BufRead};


#[derive(pest_derive::Parser)]
#[grammar = "calculator.pest"]
pub struct CalculatorParser;

lazy_static::lazy_static! {
    static ref PRATT_PARSER: PrattParser<Rule> = {
        use pest::pratt_parser::{Assoc::*, Op};
        use Rule::*;

        PrattParser::new()
            .op(Op::infix(add, Left) | Op::infix(subtract, Left))
            .op(Op::infix(multiply, Left) | Op::infix(divide, Left) | Op::infix(power, Left))
            .op(Op::prefix(unary_minus))
    };
}

#[derive(Debug)]
pub enum Expr {
    Number(f32),
    Name(String),
    UnaryMinus(Box<Expr>),
    BinOp {
        lhs: Box<Expr>,
        op: Op,
        rhs: Box<Expr>,
    },
    Function {
        name: String,
        args: Vec<Box<Expr>>
    },
}

// then try writing rust code from expr
pub fn parse_expr(pairs: Pairs<Rule>) -> Expr {
    PRATT_PARSER
        .map_primary(|primary| match primary.as_rule() {
            Rule::number => Expr::Number(primary.as_str().parse::<f32>().unwrap()),
            Rule::name => Expr::Name(String::from(primary.as_str())),
            Rule::expr => parse_expr(primary.into_inner()),
            // Rule::function => {
            //     let mut inner_rules = primary.into_inner(); // { name ~ "=" ~ value }

            //     let name: String = String::from(inner_rules.next().unwrap().as_str());
            //     // this needs to aggregate each expr for each inner
            //     // keep hitting next until there is no next, then aggregate into args
            //     let args: Expr = parse_expr(inner_rules.next().unwrap().into_inner());

            //     Expr::Function { name: name, args: args }
            // }, // get name and inner for each expr
            rule => unreachable!("Expr::parse expected atom, found {:?}", rule),
        })
        .map_infix(|lhs, op, rhs| {
            let op = match op.as_rule() {
                Rule::add => Op::Add,
                Rule::subtract => Op::Subtract,
                Rule::multiply => Op::Multiply,
                Rule::divide => Op::Divide,
                Rule::power => Op::Power,
                rule => unreachable!("Expr::parse expected infix operation, found {:?}", rule),
            };
            Expr::BinOp {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
            }
        })
        .map_prefix(|op, rhs| match op.as_rule() {
            Rule::unary_minus => Expr::UnaryMinus(Box::new(rhs)),
            _ => unreachable!(),
        })
        .parse(pairs)
}

#[derive(Debug)]
pub enum Op {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
}

impl Expr {
    pub fn to_string(&self) -> String {
        match self {
            Expr::Number(n) => n.to_string(),
            Expr::Name(name) => name.clone(),
            Expr::UnaryMinus(expr) => format!("-{}", expr.to_string()),
            Expr::BinOp { lhs, op, rhs } => {
                let op_str = match op {
                    Op::Add => "+",
                    Op::Subtract => "-",
                    Op::Multiply => "*",
                    Op::Divide => "/",
                    Op::Power => "^",
                };
                format!("({} {} {})", lhs.to_string(), op_str, rhs.to_string())
            }
            Expr::Function { name, args } => {
                format!(
                    "{}({})",
                    name, 
                    args.iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                    )
            }
        }
    }
}

fn main() -> io::Result<()> {
    for (key, value) in env::vars() {
        if key == "FOO" {
            println!("{}: {}", key, value);
        }
    }

    for line in io::stdin().lock().lines() {
        match CalculatorParser::parse(Rule::equation, &line?) {
            Ok(mut pairs) => {
                let current_expr = parse_expr(pairs.next().unwrap().into_inner());
                println!(
                    "Parsed: {:#?}\nString: {}",
                    // inner of expr
                    current_expr,
                    current_expr.to_string(),
                );
            }
            Err(e) => {
                eprintln!("Parse failed: {:?}", e);
            }
        }
    }

    Ok(())
}

// todo support functions in expression
