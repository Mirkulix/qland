//! The `net` effect: outgoing HTTP requests through the host (Phase 3 spec §3.1, §5.1).
//!
//! A request goes out only to a granted host and port (`net:<host>[:port]`). DNS names are
//! resolved by the host, every address is checked against the private and reserved ranges (one
//! blocked address blocks the name), and the connection uses exactly the checked addresses (no
//! second lookup, so DNS rebinding between check and connect is impossible). An IP literal grant
//! allows exactly that address and port. No redirects are followed, proxies are never used, and
//! connect time, total time, response size and the number of requests are bounded.

use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The host of a grant.
#[derive(Debug, Clone, PartialEq)]
pub enum NetHost {
    /// A DNS name, lowercase ASCII.
    Name(String),
    /// `*.<suffix>`: every name below `<suffix>` (not `<suffix>` itself).
    Below(String),
    /// An IP literal: exactly this address.
    Ip(IpAddr),
}

/// A `net:<host>[:port]` grant.
#[derive(Debug, Clone, PartialEq)]
pub struct NetGrant {
    pub host: NetHost,
    /// None: only the default ports (80 for http, 443 for https).
    pub port: Option<u16>,
}

/// Connect timeout of one request.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Total time of one request (never more than the execution has left).
pub const TOTAL_TIMEOUT: Duration = Duration::from_secs(10);
/// Largest accepted response body.
pub const MAX_RESPONSE_BYTES: usize = 4 << 20;
/// Headers the host sets itself; the guest may not.
const HOST_HEADERS: [&str; 5] = ["host", "content-length", "connection", "user-agent", "transfer-encoding"];

fn valid_name(n: &str) -> bool {
    let labels: Vec<&str> = n.split('.').collect();
    n.len() <= 253
        && labels.iter().all(|l| !l.is_empty() && l.len() <= 63 && !l.starts_with('-') && !l.ends_with('-') && l.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'))
        // A numeric last label makes the URL parser read the name as an IPv4 address.
        && labels.last().is_some_and(|l| !l.chars().all(|c| c.is_ascii_digit()) && !l.starts_with("0x"))
}

fn is_localhost(n: &str) -> bool {
    n == "localhost" || n.ends_with(".localhost")
}

impl NetGrant {
    /// Parses `net:<host>`, `net:<host>:<port>`, `net:*.<domain>[:port]`, `net:1.2.3.4[:port]`,
    /// `net:[::1][:port]`.
    pub fn parse(item: &str) -> Result<NetGrant, String> {
        let spec = item.strip_prefix("net:").unwrap_or("");
        if spec.is_empty() {
            return Err("`net` needs a host: `net:<host>[:port]`".into());
        }
        let (host, port) = if let Some(rest) = spec.strip_prefix('[') {
            let (ip, after) = rest.split_once(']').ok_or_else(|| format!("`{item}`: missing `]`"))?;
            let port = match after {
                "" => None,
                p => Some(p.strip_prefix(':').ok_or_else(|| format!("`{item}`: expected `:<port>` after `]`"))?),
            };
            let ip: Ipv6Addr = ip.parse().map_err(|_| format!("`{item}`: not an IPv6 address"))?;
            (NetHost::Ip(IpAddr::V6(ip)), port)
        } else {
            let (h, p) = match spec.rsplit_once(':') {
                Some((h, p)) => (h, Some(p)),
                None => (spec, None),
            };
            let host = if let Ok(ip) = h.parse::<Ipv4Addr>() {
                NetHost::Ip(IpAddr::V4(ip))
            } else if let Some(suffix) = h.strip_prefix("*.") {
                if !valid_name(suffix) || !suffix.contains('.') {
                    return Err(format!("`{item}`: `*.` must be followed by a domain with at least two labels, e.g. `net:*.example.com`"));
                }
                NetHost::Below(suffix.to_string())
            } else if valid_name(h) {
                NetHost::Name(h.to_string())
            } else {
                return Err(format!("`{item}`: `{h}` is not a lowercase DNS name (IDNA names in their `xn--` form) or an IP address"));
            };
            (host, p)
        };
        if let NetHost::Name(n) | NetHost::Below(n) = &host {
            if is_localhost(n) {
                return Err(format!("`{item}`: `localhost` names are not allowed; grant the address, e.g. `net:127.0.0.1:{}`", port.unwrap_or("8080")));
            }
        }
        let port = match port {
            None => None,
            Some(p) => Some(p.parse::<u16>().ok().filter(|p| *p > 0).ok_or_else(|| format!("`{item}`: `{p}` is not a port (1-65535)"))?),
        };
        Ok(NetGrant { host, port })
    }

    /// The grant as written.
    pub fn spec(&self) -> String {
        let host = match &self.host {
            NetHost::Name(n) => n.clone(),
            NetHost::Below(n) => format!("*.{n}"),
            NetHost::Ip(IpAddr::V4(ip)) => ip.to_string(),
            NetHost::Ip(IpAddr::V6(ip)) => format!("[{ip}]"),
        };
        match self.port {
            Some(p) => format!("net:{host}:{p}"),
            None => format!("net:{host}"),
        }
    }

    fn allows_port(&self, port: u16, default: u16) -> bool {
        match self.port {
            Some(p) => p == port,
            None => port == default && (port == 80 || port == 443),
        }
    }
}

/// An IPv4 address embedded in an IPv6 one (mapped, compatible, NAT64, 6to4), which reaches the
/// same IPv4 host.
fn embedded_v4(ip: &Ipv6Addr) -> Option<Ipv4Addr> {
    let s = ip.segments();
    let v4 = |a: u16, b: u16| Ipv4Addr::new((a >> 8) as u8, a as u8, (b >> 8) as u8, b as u8);
    if let Some(v) = ip.to_ipv4_mapped() {
        return Some(v);
    }
    match s {
        // ::ffff:0:a.b.c.d (IPv4-translated, SIIT)
        [0, 0, 0, 0, 0xffff, 0, a, b] => Some(v4(a, b)),
        // ::a.b.c.d (IPv4-compatible; :: and ::1 themselves are handled as IPv6)
        [0, 0, 0, 0, 0, 0, a, b] if (a, b) != (0, 0) && (a, b) != (0, 1) => Some(v4(a, b)),
        // 64:ff9b::/96 and 64:ff9b:1::/48 (NAT64)
        [0x64, 0xff9b, 0, 0, 0, 0, a, b] => Some(v4(a, b)),
        [0x64, 0xff9b, 1, ..] => Some(v4(s[6], s[7])),
        // 2002:AABB:CCDD::/48 (6to4)
        [0x2002, a, b, ..] => Some(v4(a, b)),
        _ => None,
    }
}

fn blocked_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    a == 0 || a == 10 || a == 127 || (a == 100 && (64..128).contains(&b)) || (a == 169 && b == 254) || (a == 172 && (16..32).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 168)
        || (a == 198 && (b == 18 || b == 19))
        || a >= 224
}

/// True if a DNS name resolving to `ip` must not be contacted: private, loopback, link-local
/// (cloud metadata), shared, benchmark, multicast and reserved ranges, also when embedded in IPv6.
pub fn blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => blocked_v4(v4),
        IpAddr::V6(v6) => {
            if let Some(v4) = embedded_v4(&v6) {
                return blocked_v4(v4);
            }
            let s = v6.segments();
            let s0 = s[0];
            v6.is_unspecified()
                || v6.is_loopback()
                || (s0 & 0xfe00) == 0xfc00 // unique local
                || (s0 & 0xffc0) == 0xfe80 // link-local
                || (s0 & 0xffc0) == 0xfec0 // site-local (deprecated)
                || (s0 & 0xff00) == 0xff00 // multicast
                || (s0 == 0x2001 && s[1] == 0) // Teredo (tunnels to IPv4 hosts)
                || (s0 == 0x0100 && s[1..4] == [0, 0, 0]) // discard prefix
        }
    }
}

/// The same address for comparing with an IP grant (IPv4-mapped IPv6 is its IPv4 address).
fn canonical_ip(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(ip),
        v4 => v4,
    }
}

/// A request from the guest (`HttpCall`).
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

/// A response (`HttpResponse`): status, headers with lowercase names, UTF-8 body.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    pub status: i64,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

/// A failed request (`NetError`).
#[derive(Debug, Clone, PartialEq)]
pub enum NetErr {
    Denied(String),
    Invalid(String),
    Connect(String),
    Timeout,
    TooLarge,
}

/// Looks up a name (in a helper thread, so a slow resolver cannot outlast `limit`).
pub type Lookup = fn(&str, u16, Duration) -> std::io::Result<Vec<SocketAddr>>;

/// The system resolver, bounded by `limit`.
pub fn system_lookup(host: &str, port: u16, limit: Duration) -> std::io::Result<Vec<SocketAddr>> {
    let (tx, rx) = mpsc::channel();
    let name = host.to_string();
    std::thread::Builder::new().name("qal-dns".into()).spawn(move || {
        let _ = tx.send((name.as_str(), port).to_socket_addrs().map(|a| a.collect::<Vec<_>>()));
    })?;
    rx.recv_timeout(limit).map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "DNS lookup timed out"))?
}

pub fn is_token(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}

/// Checks the call against the grants and resolves its destination: the URL and the addresses to
/// connect to.
pub fn plan(grants: &[NetGrant], call: &Call, lookup: Lookup, limit: Duration) -> Result<(url::Url, Vec<SocketAddr>), NetErr> {
    if !is_token(&call.method) {
        return Err(NetErr::Invalid(format!("`{}` is not an HTTP method", call.method.chars().take(40).collect::<String>())));
    }
    for (n, v) in &call.headers {
        if !is_token(n) {
            return Err(NetErr::Invalid(format!("`{}` is not a header name", n.chars().take(40).collect::<String>())));
        }
        if v.bytes().any(|b| b == b'\r' || b == b'\n' || b == 0) {
            return Err(NetErr::Invalid(format!("the value of header `{n}` contains CR, LF or NUL")));
        }
        if HOST_HEADERS.iter().any(|h| h.eq_ignore_ascii_case(n)) {
            return Err(NetErr::Invalid(format!("header `{n}` is set by the host")));
        }
    }
    let url = url::Url::parse(&call.url).map_err(|e| NetErr::Invalid(format!("invalid URL: {e}")))?;
    let default = match url.scheme() {
        "http" => 80,
        "https" => 443,
        s => return Err(NetErr::Invalid(format!("unsupported URL scheme `{s}` (only http and https)"))),
    };
    if !url.username().is_empty() || url.password().is_some() {
        return Err(NetErr::Invalid("URLs with user information (`user:pass@`) are not allowed".into()));
    }
    let port = url.port_or_known_default().unwrap_or(default);
    let host = url.host().ok_or_else(|| NetErr::Invalid("the URL has no host".into()))?;
    let shown = match &host {
        url::Host::Domain(d) => d.to_string(),
        url::Host::Ipv4(ip) => ip.to_string(),
        url::Host::Ipv6(ip) => format!("[{ip}]"),
    };
    let denied = || {
        let need = match &host {
            url::Host::Domain(d) => format!("net:{}", d.trim_end_matches('.')),
            _ => format!("net:{shown}"),
        };
        let need = if port == default { need } else { format!("{need}:{port}") };
        NetErr::Denied(format!("`{shown}:{port}` is not granted; it needs `{need}`"))
    };
    match host {
        url::Host::Ipv4(_) | url::Host::Ipv6(_) => {
            let ip = canonical_ip(match host {
                url::Host::Ipv4(v4) => IpAddr::V4(v4),
                url::Host::Ipv6(v6) => IpAddr::V6(v6),
                url::Host::Domain(_) => return Err(denied()),
            });
            let ok = grants.iter().any(|g| matches!(g.host, NetHost::Ip(gip) if canonical_ip(gip) == ip) && g.allows_port(port, default));
            if !ok {
                return Err(denied());
            }
            Ok((url, vec![SocketAddr::new(ip, port)]))
        }
        url::Host::Domain(d) => {
            let name = d.trim_end_matches('.').to_ascii_lowercase();
            let ok = grants.iter().any(|g| {
                let host_ok = match &g.host {
                    NetHost::Name(n) => *n == name,
                    NetHost::Below(s) => name.len() > s.len() + 1 && name.ends_with(s.as_str()) && name.as_bytes()[name.len() - s.len() - 1] == b'.',
                    NetHost::Ip(_) => false,
                };
                host_ok && g.allows_port(port, default)
            });
            if !ok || is_localhost(&name) {
                return Err(denied());
            }
            let addrs = lookup(&name, port, limit).map_err(|e| if e.kind() == std::io::ErrorKind::TimedOut { NetErr::Timeout } else { NetErr::Connect(format!("cannot resolve `{name}`: {e}")) })?;
            if addrs.is_empty() {
                return Err(NetErr::Connect(format!("`{name}` has no address")));
            }
            if let Some(bad) = addrs.iter().find(|a| blocked(a.ip())) {
                return Err(NetErr::Denied(format!("`{name}` resolves to the private or reserved address {}; names may only reach public addresses (grant an IP literal for local services)", bad.ip())));
            }
            Ok((url, addrs.into_iter().map(|a| SocketAddr::new(a.ip(), port)).collect()))
        }
    }
}

/// Sends the request to the planned addresses and reads the response within `limit`.
pub fn send(call: &Call, url: url::Url, addrs: Vec<SocketAddr>, limit: Duration) -> Result<Reply, NetErr> {
    let start = Instant::now();
    let total = limit.min(TOTAL_TIMEOUT);
    let fixed = addrs.clone();
    let agent = ureq::AgentBuilder::new()
        .redirects(0)
        .try_proxy_from_env(false)
        .timeout_connect(CONNECT_TIMEOUT.min(total))
        .timeout(total)
        .max_idle_connections(0)
        .user_agent(concat!("qal/", env!("CARGO_PKG_VERSION")))
        .resolver(move |_: &str| -> std::io::Result<Vec<SocketAddr>> { Ok(fixed.clone()) })
        .build();
    let mut req = agent.request_url(&call.method, &url).set("connection", "close");
    for (n, v) in &call.headers {
        req = req.set(n, v);
    }
    let resp = match req.send_bytes(call.body.as_bytes()) {
        Ok(r) | Err(ureq::Error::Status(_, r)) => r,
        Err(ureq::Error::Transport(t)) => {
            let msg = t.to_string();
            let timed_out = start.elapsed() >= total || msg.contains("timed out") || msg.contains("Timeout") || msg.contains("WouldBlock");
            return Err(match t.kind() {
                _ if timed_out => NetErr::Timeout,
                ureq::ErrorKind::InvalidUrl | ureq::ErrorKind::UnknownScheme => NetErr::Invalid(msg),
                _ => NetErr::Connect(msg),
            });
        }
    };
    let status = resp.status() as i64;
    let mut headers = Vec::new();
    for name in resp.headers_names() {
        for v in resp.all(&name) {
            headers.push((name.to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    let mut bytes = Vec::new();
    let read = resp.into_reader().take(MAX_RESPONSE_BYTES as u64 + 1).read_to_end(&mut bytes);
    if let Err(e) = read {
        return Err(if start.elapsed() >= total || e.kind() == std::io::ErrorKind::TimedOut || e.kind() == std::io::ErrorKind::WouldBlock { NetErr::Timeout } else { NetErr::Connect(format!("reading the response: {e}")) });
    }
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err(NetErr::TooLarge);
    }
    let body = String::from_utf8(bytes).map_err(|_| NetErr::Invalid("response body is not UTF-8".into()))?;
    Ok(Reply { status, headers, body })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(s: &str) -> NetGrant {
        NetGrant::parse(s).unwrap()
    }

    fn call(url: &str) -> Call {
        Call { method: "GET".into(), url: url.into(), headers: vec![], body: String::new() }
    }

    fn fake(host: &str, port: u16, _: Duration) -> std::io::Result<Vec<SocketAddr>> {
        let ips: Vec<IpAddr> = match host {
            "public.example.com" | "api.example.com" | "a.b.example.com" => vec!["93.184.216.34".parse().unwrap()],
            "rebind.example.com" => vec!["93.184.216.34".parse().unwrap(), "127.0.0.1".parse().unwrap()],
            "meta.example.com" => vec!["169.254.169.254".parse().unwrap()],
            "mapped.example.com" => vec!["::ffff:10.0.0.1".parse().unwrap()],
            "v6.example.com" => vec!["2606:2800:220:1:248:1893:25c8:1946".parse().unwrap()],
            "slow.example.com" => return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "slow")),
            _ => return Err(std::io::Error::other("no such host")),
        };
        Ok(ips.into_iter().map(|ip| SocketAddr::new(ip, port)).collect())
    }

    fn plan_with(grants: &[&str], url: &str) -> Result<Vec<SocketAddr>, NetErr> {
        let grants: Vec<NetGrant> = grants.iter().map(|s| g(s)).collect();
        plan(&grants, &call(url), fake, Duration::from_secs(1)).map(|(_, a)| a)
    }

    #[test]
    fn grants_parse() {
        assert_eq!(g("net:api.example.com"), NetGrant { host: NetHost::Name("api.example.com".into()), port: None });
        assert_eq!(g("net:api.example.com:8443").port, Some(8443));
        assert_eq!(g("net:*.example.com").host, NetHost::Below("example.com".into()));
        assert_eq!(g("net:127.0.0.1:9000"), NetGrant { host: NetHost::Ip("127.0.0.1".parse().unwrap()), port: Some(9000) });
        assert_eq!(g("net:[::1]:9000").host, NetHost::Ip("::1".parse().unwrap()));
        assert_eq!(g("net:[::1]").port, None);
        for s in ["net:[::1]:9000", "net:*.example.com:81", "net:10.1.2.3", "net:xn--bcher-kva.example"] {
            assert_eq!(g(s).spec(), s);
        }
        let e = |s: &str| NetGrant::parse(s).unwrap_err();
        assert!(e("net:").contains("needs a host"));
        assert!(e("net:localhost:8080").contains("`localhost` names are not allowed"));
        assert!(e("net:*.localhost").contains("localhost"));
        assert!(e("net:API.example.com").contains("not a lowercase DNS name"));
        assert!(e("net:2130706433").contains("not a lowercase DNS name"), "decimal IPv4 is not a name");
        assert!(e("net:0x7f.0.0.1").contains("not a lowercase DNS name"));
        assert!(e("net:010.0.0.1").contains("not a lowercase DNS name"), "octal-looking octets are refused");
        assert!(e("net:*.com").contains("at least two labels"));
        assert!(e("net:a.com:0").contains("not a port"));
        assert!(e("net:a.com:99999").contains("not a port"));
        assert!(e("net:[::1").contains("missing `]`"));
        assert!(e("net:bad_name.com").contains("not a lowercase"));
    }

    #[test]
    fn private_and_reserved_ranges_are_blocked_for_names() {
        for ip in ["0.0.0.0", "10.1.2.3", "100.64.0.1", "127.0.0.1", "169.254.169.254", "172.16.0.1", "172.31.255.255", "192.0.0.8", "192.168.1.1", "198.18.0.1", "224.0.0.1", "240.0.0.1", "255.255.255.255", "::", "::1", "fc00::1", "fd12::1", "fe80::1", "ff02::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1", "64:ff9b::a9fe:a9fe", "::7f00:1", "2002:a9fe:a9fe::1", "64:ff9b:1::7f00:1"] {
            assert!(blocked(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["::ffff:0:7f00:1", "::ffff:0:a9fe:a9fe", "fec0::1", "2001:0:4136:e378:8000:63bf:80ff:fffe", "100::1"] {
            assert!(blocked(ip.parse().unwrap()), "{ip} (found in review)");
        }
        for ip in ["93.184.216.34", "8.8.8.8", "172.32.0.1", "100.128.0.1", "2606:4700::1111", "::ffff:8.8.8.8", "2002:808:808::1", "2001:db8::1"] {
            assert!(!blocked(ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn destinations_must_be_granted_and_public() {
        assert_eq!(plan_with(&["net:api.example.com"], "https://api.example.com/v1?x=1").unwrap(), vec!["93.184.216.34:443".parse().unwrap()]);
        assert!(plan_with(&["net:api.example.com"], "http://api.example.com/").is_ok(), "port 80 for http");
        assert!(matches!(plan_with(&["net:api.example.com"], "http://api.example.com:8080/"), Err(NetErr::Denied(m)) if m.contains("net:api.example.com:8080")));
        assert!(plan_with(&["net:api.example.com:8080"], "http://api.example.com:8080/").is_ok());
        assert!(matches!(plan_with(&["net:api.example.com:8080"], "http://api.example.com/"), Err(NetErr::Denied(_))));
        assert!(plan_with(&["net:*.example.com"], "https://a.b.example.com/").is_ok());
        assert!(matches!(plan_with(&["net:*.example.com"], "https://example.com/"), Err(NetErr::Denied(_))), "not the domain itself");
        assert!(matches!(plan_with(&["net:*.example.com"], "https://evilexample.com/"), Err(NetErr::Denied(_))));
        assert!(plan_with(&["net:api.example.com"], "https://API.Example.COM./").is_ok(), "the URL parser lowercases; a trailing dot is the same name");
        // DNS answers with any private address are refused (rebinding, cloud metadata, mapped IPv6).
        for host in ["rebind", "meta", "mapped"] {
            let url = format!("https://{host}.example.com/");
            assert!(matches!(plan_with(&[&format!("net:{host}.example.com")], &url), Err(NetErr::Denied(m)) if m.contains("private or reserved")), "{host}");
        }
        assert!(plan_with(&["net:v6.example.com"], "https://v6.example.com/").is_ok());
        assert_eq!(plan_with(&["net:slow.example.com"], "https://slow.example.com/"), Err(NetErr::Timeout));
        assert!(matches!(plan_with(&["net:nowhere.example.com"], "https://nowhere.example.com/"), Err(NetErr::Connect(_))));
        // IP literals need an IP grant; every spelling of an address is that address.
        assert_eq!(plan_with(&["net:127.0.0.1:9000"], "http://127.0.0.1:9000/").unwrap(), vec!["127.0.0.1:9000".parse().unwrap()]);
        for url in ["http://2130706433:9000/", "http://0x7f000001:9000/", "http://017700000001:9000/", "http://127.1:9000/", "http://[::ffff:127.0.0.1]:9000/"] {
            assert_eq!(plan_with(&["net:127.0.0.1:9000"], url).unwrap(), vec!["127.0.0.1:9000".parse().unwrap()], "{url}");
            assert!(matches!(plan_with(&["net:api.example.com"], url), Err(NetErr::Denied(_))), "{url}");
        }
        assert!(matches!(plan_with(&["net:127.0.0.1:9000"], "http://127.0.0.1:9001/"), Err(NetErr::Denied(_))));
        assert!(plan_with(&["net:[::1]:9000"], "http://[::1]:9000/").is_ok());
        assert!(matches!(plan_with(&[], "http://localhost/"), Err(NetErr::Denied(_))));
    }

    #[test]
    fn malformed_calls_are_invalid() {
        let grants = vec![g("net:api.example.com")];
        let inv = |c: Call| match plan(&grants, &c, fake, Duration::from_secs(1)) {
            Err(NetErr::Invalid(m)) => m,
            other => panic!("{other:?}"),
        };
        assert!(inv(call("ftp://api.example.com/")).contains("unsupported URL scheme"));
        assert!(inv(call("file:///etc/passwd")).contains("unsupported URL scheme"));
        assert!(inv(call("https://user:pw@api.example.com/")).contains("user information"));
        assert!(inv(call("https://user@api.example.com/")).contains("user information"));
        assert!(inv(call("not a url")).contains("invalid URL"));
        assert!(inv(Call { method: "GE T".into(), ..call("https://api.example.com/") }).contains("not an HTTP method"));
        for h in ["host", "Content-Length", "connection", "User-Agent", "transfer-encoding"] {
            assert!(inv(Call { headers: vec![(h.into(), "x".into())], ..call("https://api.example.com/") }).contains("set by the host"), "{h}");
        }
        assert!(inv(Call { headers: vec![("x-a".into(), "v\r\nInjected: 1".into())], ..call("https://api.example.com/") }).contains("CR, LF or NUL"));
        assert!(inv(Call { headers: vec![("bad name".into(), "v".into())], ..call("https://api.example.com/") }).contains("not a header name"));
    }

    #[test]
    fn a_silent_server_times_out_and_tls_is_required_for_https() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let hold = std::thread::spawn(move || {
            let held: Vec<_> = l.incoming().take(2).collect();
            std::thread::sleep(Duration::from_secs(2));
            drop(held);
        });
        let grants = vec![g(&format!("net:127.0.0.1:{}", addr.port()))];
        let t = Instant::now();
        let c = call(&format!("http://127.0.0.1:{}/", addr.port()));
        let (url, addrs) = plan(&grants, &c, fake, Duration::from_secs(1)).unwrap();
        assert_eq!(send(&c, url, addrs, Duration::from_millis(300)), Err(NetErr::Timeout));
        assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
        // https to a server that does not speak TLS fails (certificates are always verified).
        let c = call(&format!("https://127.0.0.1:{}/", addr.port()));
        let (url, addrs) = plan(&grants, &c, fake, Duration::from_secs(1)).unwrap();
        assert!(matches!(send(&c, url, addrs, Duration::from_millis(300)), Err(NetErr::Timeout | NetErr::Connect(_))));
        hold.join().unwrap();
    }
}
