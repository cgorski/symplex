//! `#[derive(NodeShape)]`: the shape-dependent methods of symplex's
//! expression node enum, generated from its declaration.
//!
//! Which operands of a node are children, how many there are, how to
//! rebuild the node with other children, how to move it to another arena
//! and which operand a binder binds are pure functions of each variant's
//! field types.  Written by hand they were five ~100-line matches over the
//! ~90 variants that had to be kept in step (and had drifted: a rebuild
//! that forgot a canonical constructor).  The derive reads the field types:
//!
//! | field type | role |
//! |---|---|
//! | `ExprId` | one child |
//! | `SmallVec<[ExprId; N]>` | a list of children |
//! | `SmallVec<[(ExprId, ExprId); N]>` | a list of child pairs (flattened in order) |
//! | `#[payload] ExprId` | an id that is not a child (a physical constant's value): remapped, never walked |
//! | `NumId`, `SymbolId` | side-table references: remapped |
//! | anything else | a `Copy` payload, carried over unchanged |
//!
//! A variant without children is an atom.  Two variant attributes carry the
//! binder tables (operand positions, counted over all fields):
//!
//! * `#[binder(var = 1, scoped(0), outer(2, 3))]` — the node binds the
//!   operand `var` in the `scoped` operands; the `outer` operands are in the
//!   enclosing scope;
//! * `#[var_slot(var = 1, scoped(0), outer(2, 3))]` — the node is a
//!   function *of* its variable (`Derivative`), same layout.
//!
//! Every field of a binder or variable-slot variant must be an `ExprId`
//! child, named exactly once by `var`, `scoped` or `outer`.
//!
//! Generated, on the enum (the names `ExprId`, `NumId`, `SymbolId`,
//! `Binder` and `IdRemap` are resolved where the enum is declared):
//! `children`, `for_each_child`, `child_count`, `is_atom` (`pub`);
//! `map_children`, `remap_ids`, `binder_operands`, `var_slot_operands`,
//! `with_binder_operands`, `with_var_slot_operands`, `var_operand`,
//! `variant_name` and the `VARIANT_NAMES` table (`pub(crate)`).

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Fields, Ident, Type};

/// What one field of a variant is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Child,
    ChildList,
    ChildPairs,
    PayloadExpr,
    Num,
    Symbol,
    Payload,
}

/// Operand positions of a binder or variable slot.
struct Slots {
    var: usize,
    scoped: Vec<usize>,
    outer: Vec<usize>,
}

struct VariantShape {
    ident: Ident,
    /// `None` for a unit variant.
    fields: Option<Vec<(Role, Type)>>,
    binder: Option<Slots>,
    var_slot: Option<Slots>,
}

impl VariantShape {
    fn roles(&self) -> impl Iterator<Item = Role> + '_ {
        self.fields.iter().flatten().map(|(r, _)| *r)
    }

    fn is_atom(&self) -> bool {
        !self
            .roles()
            .any(|r| matches!(r, Role::Child | Role::ChildList | Role::ChildPairs))
    }

    fn arity(&self) -> usize {
        self.fields.as_ref().map_or(0, Vec::len)
    }

    /// `Self::V`, `Self::V(..)` or `Self::V(f0, _, f2)` binding the fields
    /// for which `used` holds.
    fn pattern(&self, used: impl Fn(usize, Role) -> bool) -> TokenStream2 {
        let v = &self.ident;
        match &self.fields {
            None => quote!(Self::#v),
            Some(fields) => {
                let binds = fields.iter().enumerate().map(|(i, (role, _))| {
                    if used(i, *role) {
                        let f = field_ident(i);
                        quote!(#f)
                    } else {
                        quote!(_)
                    }
                });
                quote!(Self::#v(#(#binds),*))
            }
        }
    }

    fn wild_pattern(&self) -> TokenStream2 {
        let v = &self.ident;
        match &self.fields {
            None => quote!(Self::#v),
            Some(_) => quote!(Self::#v(..)),
        }
    }
}

fn field_ident(i: usize) -> Ident {
    format_ident!("f{}", i)
}

fn is_child_role(r: Role) -> bool {
    matches!(r, Role::Child | Role::ChildList | Role::ChildPairs)
}

/// The last path segment of a type, if it is a plain path type.
fn last_segment(ty: &Type) -> Option<&syn::PathSegment> {
    match ty {
        Type::Path(p) if p.qself.is_none() => p.path.segments.last(),
        _ => None,
    }
}

fn is_named(ty: &Type, name: &str) -> bool {
    last_segment(ty).is_some_and(|s| s.ident == name && s.arguments.is_none())
}

/// `SmallVec<[T; N]>` → `T`.
fn smallvec_elem(ty: &Type) -> Option<&Type> {
    let seg = last_segment(ty)?;
    if seg.ident != "SmallVec" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    match args.args.first()? {
        syn::GenericArgument::Type(Type::Array(arr)) => Some(&arr.elem),
        _ => None,
    }
}

fn classify(field: &syn::Field) -> syn::Result<Role> {
    let payload = field.attrs.iter().any(|a| a.path().is_ident("payload"));
    let ty = &field.ty;
    if payload {
        return if is_named(ty, "ExprId") {
            Ok(Role::PayloadExpr)
        } else {
            Err(syn::Error::new(
                field.span(),
                "#[payload] marks an `ExprId` field that is not a child",
            ))
        };
    }
    if is_named(ty, "ExprId") {
        return Ok(Role::Child);
    }
    if is_named(ty, "NumId") {
        return Ok(Role::Num);
    }
    if is_named(ty, "SymbolId") {
        return Ok(Role::Symbol);
    }
    if let Some(elem) = smallvec_elem(ty) {
        if is_named(elem, "ExprId") {
            return Ok(Role::ChildList);
        }
        if let Type::Tuple(t) = elem
            && t.elems.len() == 2
            && t.elems.iter().all(|e| is_named(e, "ExprId"))
        {
            return Ok(Role::ChildPairs);
        }
    }
    Ok(Role::Payload)
}

/// Parses `var = 1, scoped(0), outer(2, 3)`.
fn parse_slots(attr: &syn::Attribute) -> syn::Result<Slots> {
    let mut var = None;
    let mut scoped = Vec::new();
    let mut outer = Vec::new();
    attr.parse_nested_meta(|meta| {
        let list = |meta: &syn::meta::ParseNestedMeta| -> syn::Result<Vec<usize>> {
            let content;
            syn::parenthesized!(content in meta.input);
            let lits =
                syn::punctuated::Punctuated::<syn::LitInt, syn::Token![,]>::parse_terminated(
                    &content,
                )?;
            lits.iter().map(syn::LitInt::base10_parse).collect()
        };
        if meta.path.is_ident("var") {
            let lit: syn::LitInt = meta.value()?.parse()?;
            var = Some(lit.base10_parse()?);
            Ok(())
        } else if meta.path.is_ident("scoped") {
            scoped = list(&meta)?;
            Ok(())
        } else if meta.path.is_ident("outer") {
            outer = list(&meta)?;
            Ok(())
        } else {
            Err(meta.error("expected `var = N`, `scoped(..)` or `outer(..)`"))
        }
    })?;
    let var = var.ok_or_else(|| syn::Error::new(attr.span(), "missing `var = N`"))?;
    Ok(Slots { var, scoped, outer })
}

fn check_slots(v: &VariantShape, s: &Slots, attr: &syn::Attribute) -> syn::Result<()> {
    let n = v.arity();
    let mut seen = vec![0usize; n];
    for &i in std::iter::once(&s.var).chain(&s.scoped).chain(&s.outer) {
        match seen.get_mut(i) {
            Some(c) => *c += 1,
            None => {
                return Err(syn::Error::new(
                    attr.span(),
                    format!("operand {i} out of range (the variant has {n} fields)"),
                ));
            }
        }
    }
    if seen.iter().any(|&c| c != 1) {
        return Err(syn::Error::new(
            attr.span(),
            "every operand must be named exactly once by `var`, `scoped` or `outer`",
        ));
    }
    if v.roles().any(|r| r != Role::Child) {
        return Err(syn::Error::new(
            attr.span(),
            "a binder's operands must all be `ExprId` children",
        ));
    }
    Ok(())
}

fn parse_variant(variant: &syn::Variant) -> syn::Result<VariantShape> {
    let fields = match &variant.fields {
        Fields::Unit => None,
        Fields::Unnamed(f) => Some(
            f.unnamed
                .iter()
                .map(|field| Ok((classify(field)?, field.ty.clone())))
                .collect::<syn::Result<Vec<_>>>()?,
        ),
        Fields::Named(f) => {
            return Err(syn::Error::new(
                f.span(),
                "NodeShape supports unit and tuple variants only",
            ));
        }
    };
    let mut shape = VariantShape {
        ident: variant.ident.clone(),
        fields,
        binder: None,
        var_slot: None,
    };
    for attr in &variant.attrs {
        if attr.path().is_ident("binder") {
            let s = parse_slots(attr)?;
            check_slots(&shape, &s, attr)?;
            shape.binder = Some(s);
        } else if attr.path().is_ident("var_slot") {
            let s = parse_slots(attr)?;
            check_slots(&shape, &s, attr)?;
            shape.var_slot = Some(s);
        }
    }
    Ok(shape)
}

pub(crate) fn derive(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new(
            input.span(),
            "NodeShape derives on an enum",
        ));
    };
    let variants = data
        .variants
        .iter()
        .map(parse_variant)
        .collect::<syn::Result<Vec<_>>>()?;
    let name = &input.ident;

    let children = gen_children(&variants);
    let for_each = gen_for_each_child(&variants);
    let count = gen_child_count(&variants);
    let atom = gen_is_atom(&variants);
    let map = gen_map_children(&variants);
    let remap = gen_remap_ids(&variants);
    let binder = gen_slots_view(
        &variants,
        |v| v.binder.as_ref(),
        &format_ident!("binder_operands"),
    );
    let var_slot = gen_slots_view(
        &variants,
        |v| v.var_slot.as_ref(),
        &format_ident!("var_slot_operands"),
    );
    let with_binder = gen_with_slots(
        &variants,
        |v| v.binder.as_ref(),
        &format_ident!("with_binder_operands"),
    );
    let with_var_slot = gen_with_slots(
        &variants,
        |v| v.var_slot.as_ref(),
        &format_ident!("with_var_slot_operands"),
    );
    let var_operand = gen_var_operand(&variants);
    let names = gen_names(&variants);

    Ok(quote! {
        impl #name {
            #children
            #for_each
            #count
            #atom
            #map
            #remap
            #binder
            #var_slot
            #with_binder
            #with_var_slot
            #var_operand
            #names
        }
    })
}

fn gen_children(variants: &[VariantShape]) -> TokenStream2 {
    let arms = variants.iter().map(|v| {
        if v.is_atom() {
            let pat = v.wild_pattern();
            return quote!(#pat => ::smallvec::SmallVec::new(),);
        }
        let pat = v.pattern(|_, r| is_child_role(r));
        let kids: Vec<(usize, Role)> = v
            .roles()
            .enumerate()
            .filter(|&(_, r)| is_child_role(r))
            .collect();
        // The common shapes are built in one step (a hot path: canonical
        // comparison, tree walks).
        if kids.iter().all(|&(_, r)| r == Role::Child) {
            let fs = kids.iter().map(|&(i, _)| field_ident(i));
            return quote!(#pat => ::smallvec::smallvec![#(*#fs),*],);
        }
        if let [(i, Role::ChildList)] = kids.as_slice() {
            let f = field_ident(*i);
            return quote!(#pat => ::smallvec::SmallVec::from_slice(#f),);
        }
        let pushes = kids.iter().map(|&(i, r)| {
            let f = field_ident(i);
            match r {
                Role::ChildList => quote!(out.extend_from_slice(#f);),
                Role::ChildPairs => quote! {
                    for &(a, b) in #f.iter() {
                        out.push(a);
                        out.push(b);
                    }
                },
                _ => quote!(out.push(*#f);),
            }
        });
        quote! {
            #pat => {
                let mut out: ::smallvec::SmallVec<[ExprId; 6]> = ::smallvec::SmallVec::new();
                #(#pushes)*
                out
            }
        }
    });
    quote! {
        /// Returns all child `ExprId`s contained in this node.
        ///
        /// For atoms (numbers, symbols, constants, and special values) the
        /// returned collection is empty.  For compound nodes the children are
        /// returned in the order they appear in the variant (the pairs of a
        /// `Piecewise` flattened as value, condition, value, …).
        pub fn children(&self) -> ::smallvec::SmallVec<[ExprId; 6]> {
            match self {
                #(#arms)*
            }
        }
    }
}

fn gen_for_each_child(variants: &[VariantShape]) -> TokenStream2 {
    let arms = variants.iter().map(|v| {
        if v.is_atom() {
            let pat = v.wild_pattern();
            return quote!(#pat => {});
        }
        let pat = v.pattern(|_, r| is_child_role(r));
        let calls = v.roles().enumerate().map(|(i, r)| {
            let f = field_ident(i);
            match r {
                Role::Child => quote!(f(*#f);),
                Role::ChildList => quote! {
                    for &c in #f.iter() {
                        f(c);
                    }
                },
                Role::ChildPairs => quote! {
                    for &(a, b) in #f.iter() {
                        f(a);
                        f(b);
                    }
                },
                _ => quote!(),
            }
        });
        quote!(#pat => { #(#calls)* })
    });
    quote! {
        /// Calls `f` for each child `ExprId` without allocating, in the order
        /// of [`children()`](Self::children).
        ///
        /// This is the zero-allocation alternative to [`children()`](Self::children).
        /// Prefer this in hot paths (tree walks, display) where the allocation
        /// from `children()` would be significant.
        #[inline]
        pub fn for_each_child(&self, mut f: impl FnMut(ExprId)) {
            match self {
                #(#arms)*
            }
        }
    }
}

fn gen_child_count(variants: &[VariantShape]) -> TokenStream2 {
    let arms = variants.iter().map(|v| {
        let fixed = v.roles().filter(|&r| r == Role::Child).count();
        let dynamic = v
            .roles()
            .any(|r| matches!(r, Role::ChildList | Role::ChildPairs));
        if !dynamic {
            let pat = v.wild_pattern();
            return quote!(#pat => #fixed,);
        }
        let pat = v.pattern(|_, r| matches!(r, Role::ChildList | Role::ChildPairs));
        let terms = v.roles().enumerate().filter_map(|(i, r)| {
            let f = field_ident(i);
            match r {
                Role::ChildList => Some(quote!(#f.len())),
                Role::ChildPairs => Some(quote!(#f.len() * 2)),
                _ => None,
            }
        });
        if fixed == 0 {
            quote!(#pat => 0 #(+ #terms)*,)
        } else {
            quote!(#pat => #fixed #(+ #terms)*,)
        }
    });
    quote! {
        /// Returns the number of children without allocating.
        #[inline]
        pub fn child_count(&self) -> usize {
            match self {
                #(#arms)*
            }
        }
    }
}

fn gen_is_atom(variants: &[VariantShape]) -> TokenStream2 {
    let atoms = variants
        .iter()
        .filter(|v| v.is_atom())
        .map(VariantShape::wild_pattern);
    quote! {
        /// Returns `true` if this node is an atom: a variant without child
        /// expressions (numbers, symbols, the constants — including a
        /// `PhysicalConstant`, whose value is not a child —, the special
        /// values, the boolean atoms and the set atoms).
        #[inline]
        pub fn is_atom(&self) -> bool {
            matches!(self, #(#atoms)|*)
        }
    }
}

fn gen_map_children(variants: &[VariantShape]) -> TokenStream2 {
    let arms = variants.iter().map(|v| {
        if v.is_atom() {
            let pat = v.wild_pattern();
            return quote!(#pat => None,);
        }
        let pat = v.pattern(|_, _| true);
        let fields = v.fields.as_deref().unwrap_or_default();
        let lets = fields.iter().enumerate().map(|(i, (r, ty))| {
            let f = field_ident(i);
            let n = format_ident!("n{}", i);
            match r {
                Role::Child => quote!(let #n = f(*#f);),
                Role::ChildList => quote!(let #n: #ty = #f.iter().map(|&c| f(c)).collect();),
                Role::ChildPairs => {
                    quote!(let #n: #ty = #f.iter().map(|&(a, b)| (f(a), f(b))).collect();)
                }
                _ => quote!(let #n = *#f;),
            }
        });
        let same = fields.iter().enumerate().filter_map(|(i, (r, _))| {
            let f = field_ident(i);
            let n = format_ident!("n{}", i);
            match r {
                Role::Child => Some(quote!(#n == *#f)),
                Role::ChildList | Role::ChildPairs => Some(quote!(#n == *#f)),
                _ => None,
            }
        });
        let ns = (0..fields.len()).map(|i| format_ident!("n{}", i));
        let id = &v.ident;
        quote! {
            #pat => {
                #(#lets)*
                if #(#same)&&* {
                    None
                } else {
                    Some(Self::#id(#(#ns),*))
                }
            }
        }
    });
    quote! {
        /// The node with every child `c` replaced by `f(c)` (in the order of
        /// [`children()`](Self::children)), or `None` when no child changed
        /// (and for an atom).  The result is a raw node: build it with
        /// `Arena::make`, which applies the variant's canonical constructor.
        pub(crate) fn map_children(&self, mut f: impl FnMut(ExprId) -> ExprId) -> Option<Self> {
            match self {
                #(#arms)*
            }
        }
    }
}

fn gen_remap_ids(variants: &[VariantShape]) -> TokenStream2 {
    let arms = variants.iter().map(|v| {
        let id = &v.ident;
        let Some(fields) = &v.fields else {
            return quote!(Self::#id => Self::#id,);
        };
        let pat = v.pattern(|_, _| true);
        let exprs = fields.iter().enumerate().map(|(i, (r, ty))| {
            let f = field_ident(i);
            match r {
                Role::Child | Role::PayloadExpr => quote!(m.expr(*#f)),
                Role::ChildList => quote!(#f.iter().map(|&c| m.expr(c)).collect::<#ty>()),
                Role::ChildPairs => {
                    quote!(#f.iter().map(|&(a, b)| (m.expr(a), m.expr(b))).collect::<#ty>())
                }
                Role::Num => quote!(m.num(*#f)),
                Role::Symbol => quote!(m.symbol(*#f)),
                Role::Payload => quote!(*#f),
            }
        });
        // Evaluate the fields left to right into locals, then build.
        let ns: Vec<Ident> = (0..fields.len()).map(|i| format_ident!("n{}", i)).collect();
        quote! {
            #pat => {
                #(let #ns = #exprs;)*
                Self::#id(#(#ns),*)
            }
        }
    });
    quote! {
        /// The node with every id it refers to — children, a physical
        /// constant's value, numbers, symbols — mapped through `m`, field by
        /// field from the left (compaction: moving a node to another arena).
        pub(crate) fn remap_ids<M: IdRemap + ?Sized>(&self, m: &mut M) -> Self {
            match self {
                #(#arms)*
            }
        }
    }
}

fn gen_slots_view(
    variants: &[VariantShape],
    slots: impl Fn(&VariantShape) -> Option<&Slots>,
    fn_name: &Ident,
) -> TokenStream2 {
    let arms = variants.iter().map(|v| match slots(v) {
        None => {
            let pat = v.wild_pattern();
            quote!(#pat => None,)
        }
        Some(s) => {
            let pat = v.pattern(|_, _| true);
            let var = field_ident(s.var);
            let scoped = s.scoped.iter().map(|&i| field_ident(i));
            let outer = s.outer.iter().map(|&i| field_ident(i));
            quote! {
                #pat => Some(Binder {
                    var: *#var,
                    scoped: ::smallvec::smallvec![#(*#scoped),*],
                    outer: ::smallvec::smallvec![#(*#outer),*],
                }),
            }
        }
    });
    let doc = if fn_name == "binder_operands" {
        quote! {
            /// The binder layout of this node — the bound variable operand,
            /// the operands it is bound in, the operands of the enclosing
            /// scope (each in node order) — from the variant's
            /// `#[binder(..)]` attribute; `None` for a node that binds
            /// nothing.  The variable operand is not checked to be a symbol
            /// (`walk::binder` does that).
        }
    } else {
        quote! {
            /// The variable-slot layout of this node (a function *of* its
            /// variable: `Derivative`, `Integral`, `Series`, `DSolve`) from
            /// the variant's `#[var_slot(..)]` attribute; `None` for any
            /// other node.  The variable operand is not checked to be a
            /// symbol (`walk::var_slot` does that).
        }
    };
    quote! {
        #doc
        pub(crate) fn #fn_name(&self) -> Option<Binder> {
            match self {
                #(#arms)*
            }
        }
    }
}

fn gen_with_slots(
    variants: &[VariantShape],
    slots: impl Fn(&VariantShape) -> Option<&Slots>,
    fn_name: &Ident,
) -> TokenStream2 {
    let arms = variants.iter().map(|v| {
        let pat = v.wild_pattern();
        match slots(v) {
            None => quote!(#pat => None,),
            Some(s) => {
                let n = v.arity();
                let mut args: Vec<TokenStream2> = vec![quote!(var); n];
                let ss: Vec<Ident> = (0..s.scoped.len())
                    .map(|j| format_ident!("s{}", j))
                    .collect();
                let os: Vec<Ident> = (0..s.outer.len())
                    .map(|j| format_ident!("o{}", j))
                    .collect();
                for (j, &i) in s.scoped.iter().enumerate() {
                    let s = &ss[j];
                    if let Some(a) = args.get_mut(i) {
                        *a = quote!(#s);
                    }
                }
                for (j, &i) in s.outer.iter().enumerate() {
                    let o = &os[j];
                    if let Some(a) = args.get_mut(i) {
                        *a = quote!(#o);
                    }
                }
                let id = &v.ident;
                quote! {
                    #pat => match (scoped, outer) {
                        (&[#(#ss),*], &[#(#os),*]) => Some(Self::#id(#(#args),*)),
                        _ => None,
                    },
                }
            }
        }
    });
    let what = if fn_name == "with_binder_operands" {
        quote!(binder_operands)
    } else {
        quote!(var_slot_operands)
    };
    let doc = format!(
        " A node of this variant with variable operand `var` and the given \
         `scoped` / `outer` operands, in the order [`{what}`](Self::{what}) \
         lists them; `None` when the variant has no such layout or the slice \
         lengths do not match it.  The result is a raw node: build it with \
         `Arena::make`."
    );
    quote! {
        #[doc = #doc]
        pub(crate) fn #fn_name(
            &self,
            var: ExprId,
            scoped: &[ExprId],
            outer: &[ExprId],
        ) -> Option<Self> {
            match self {
                #(#arms)*
            }
        }
    }
}

fn gen_var_operand(variants: &[VariantShape]) -> TokenStream2 {
    let arms = variants
        .iter()
        .map(|v| match v.binder.as_ref().or(v.var_slot.as_ref()) {
            None => {
                let pat = v.wild_pattern();
                quote!(#pat => None,)
            }
            Some(s) => {
                let var = s.var;
                let pat = v.pattern(|i, _| i == var);
                let f = field_ident(var);
                quote!(#pat => Some(*#f),)
            }
        });
    quote! {
        /// The variable operand of a binder or variable-slot node, whether
        /// or not it is a symbol; `None` for any other node.
        pub(crate) fn var_operand(&self) -> Option<ExprId> {
            match self {
                #(#arms)*
            }
        }
    }
}

fn gen_names(variants: &[VariantShape]) -> TokenStream2 {
    let arms = variants.iter().map(|v| {
        let pat = v.wild_pattern();
        let s = v.ident.to_string();
        quote!(#pat => #s,)
    });
    let all = variants.iter().map(|v| v.ident.to_string());
    quote! {
        /// The variant's name (`"Add"`, `"Eq_"`), as `Debug` prints it.
        #[inline]
        pub(crate) fn variant_name(&self) -> &'static str {
            match self {
                #(#arms)*
            }
        }

        /// The names of all variants, in declaration order (exhaustiveness
        /// checks in tests).
        #[allow(dead_code)]
        pub(crate) const VARIANT_NAMES: &'static [&'static str] = &[#(#all),*];
    }
}
