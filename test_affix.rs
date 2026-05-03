fn strip_affix_safe<'a, T: Eq>(mut a: &'a [T], mut b: &'a [T]) -> (&'a [T], &'a [T]) {
    let mut prefix = 0;
    let len = usize::min(a.len(), b.len());
    while prefix < len && a[prefix] == b[prefix] {
        prefix += 1;
    }
    a = &a[prefix..];
    b = &b[prefix..];
    let mut suffix = 0;
    let len = usize::min(a.len(), b.len());
    while suffix < len && a[a.len() - 1 - suffix] == b[b.len() - 1 - suffix] {
        suffix += 1;
    }
    (&a[..a.len() - suffix], &b[..b.len() - suffix])
}

fn strip_affix_unsafe<'a, T: Eq>(mut a: &'a [T], mut b: &'a [T]) -> (&'a [T], &'a [T]) {
    let mut prefix = 0;
    let len = usize::min(a.len(), b.len());
    while prefix < len && unsafe { a.get_unchecked(prefix) == b.get_unchecked(prefix) } {
        prefix += 1;
    }
    a = &a[prefix..];
    b = &b[prefix..];
    let mut suffix = 0;
    let len = usize::min(a.len(), b.len());
    while suffix < len
        && unsafe { a.get_unchecked(a.len() - 1 - suffix) == b.get_unchecked(b.len() - 1 - suffix) }
    {
        suffix += 1;
    }
    (&a[..a.len() - suffix], &b[..b.len() - suffix])
}
