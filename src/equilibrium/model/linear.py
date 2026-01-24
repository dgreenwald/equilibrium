import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..solvers import perturb

logger = logging.getLogger(__name__)


class LinearModel:
    """
    Linearized representation of a nonlinear :class:`~equilibrium.model.Model`.

    The class stores the reduced-form transition matrices and provides helper
    routines for simulation and impulse-response analysis once the parent model
    has been linearized.

    Parameters
    ----------
    model : Model
        Source model that supplies steady-state derivatives.
    Phi : array_like, optional
        Autoregressive dynamics for exogenous shocks. If ``None`` the value
        must be provided during :meth:`linearize`.
    impact_matrix : array_like, optional
        Impact matrix that maps exogenous shocks into the state vector. If
        ``None`` the value must be provided during :meth:`linearize`.
    """

    def __init__(self, model, Phi=None, impact_matrix=None):
        """Initialise container with references to the parent model."""
        self.model = model
        self.A_s = None
        self.B_s = None
        self.A = None
        self.B = None
        self.G_x = None
        self.G_z = None
        self.H_x = None
        self.H_z = None
        self.Phi = Phi
        self.impact_matrix = impact_matrix
        self.irfs = None

    def linearize(self, Phi=None, impact_matrix=None, method="klein"):
        """
        Linearize the model around the steady state.

        Parameters
        ----------
        Phi : array_like, optional
            The steady state transition matrix. The default is None.
        impact_matrix : array_like, optional
            The steady state impact matrix. The default is None.
        method : str, optional
            The method to use for linearization. The default is "klein".
        """

        if Phi is not None:
            self.Phi = Phi

        if impact_matrix is not None:
            self.impact_matrix = impact_matrix

        m = self.model
        m.steady_state_derivatives()

        if method == "klein":
            self.linearize_klein()
        elif method == "aim":
            self.linearize_aim()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Intermediate variables (core only)
        J = np.hstack(
            tuple(
                m.derivatives["intermediates_core"][var] for var in ["u", "x", "z"]
            )
        )

        # Expectations vars
        dE_t = np.hstack(
            tuple(m.derivatives["expectations"][var] for var in ["u", "x", "z"])
        )

        dE_new = np.hstack(
            tuple(
                m.derivatives["expectations"][var + "_new"] for var in ["u", "x", "z"]
            )
        )

        K = dE_t + dE_new @ self.A_s

        L_s = np.hstack(
            tuple(
                m.derivatives["expectations_variables_core"][var]
                for var in ["u", "x", "z"]
            )
        )

        L_E = m.derivatives["expectations_variables_core"]["E"]

        L = L_s + L_E @ K

        # Combine
        JKL = np.vstack((J, K, L))

        N_extra = JKL.shape[0]
        Ns = self.A_s.shape[0]

        self.A = np.vstack(
            (
                np.hstack((self.A_s, np.zeros((Ns, N_extra)))),
                np.hstack((JKL @ self.A_s, np.zeros((N_extra, N_extra)))),
            )
        )

        self.B = np.vstack(
            (
                self.B_s,
                JKL @ self.B_s,
            )
        )

    def linearize_klein(self):
        """
        Linearize the model around the steady state using the Klein method.
        """

        m = self.model
        # N_tot = m.N['u'] + m.N['x']
        N_shock = self.impact_matrix.shape[1]

        f_E = m.derivatives["optimality"]["E"]

        A_klein = np.vstack(
            (
                np.hstack(
                    (
                        np.eye(m.N["x"]),
                        np.zeros((m.N["x"], m.N["u"])),
                    )
                ),
                np.hstack(
                    (
                        f_E @ m.derivatives["expectations"]["x_new"],
                        f_E @ m.derivatives["expectations"]["u_new"],
                    )
                ),
            )
        )

        B_klein = np.vstack(
            (
                np.hstack(
                    (
                        m.derivatives["transition"]["x"],
                        m.derivatives["transition"]["u"],
                    )
                ),
                np.hstack(
                    (
                        -(
                            m.derivatives["optimality"]["x"]
                            + f_E @ m.derivatives["expectations"]["x"]
                        ),
                        -(
                            m.derivatives["optimality"]["u"]
                            + f_E @ m.derivatives["expectations"]["u"]
                        ),
                    )
                ),
            )
        )

        C_klein = np.vstack(
            (
                m.derivatives["transition"]["z"],
                -(
                    m.derivatives["optimality"]["z"]
                    + f_E
                    @ (
                        m.derivatives["expectations"]["z"]
                        + m.derivatives["expectations"]["z_new"] @ self.Phi
                    )
                ),
            )
        )

        self.G_x, self.H_x, self.G_z, self.H_z = perturb.solve_klein(
            A_klein, B_klein, C_klein, self.Phi, m.N["x"]
        )

        G = np.hstack((self.G_x, self.G_z))
        H = np.hstack((self.H_x, self.H_z))

        A_xz = np.vstack((H, np.hstack((np.zeros((m.N["z"], m.N["x"])), self.Phi))))

        B_xz = np.vstack(
            (
                np.zeros((m.N["x"], N_shock)),
                self.impact_matrix,
            )
        )

        self.A_s = np.vstack(
            (
                np.hstack((np.zeros((m.N["u"], m.N["u"])), G @ A_xz)),
                np.hstack((np.zeros((m.N["x"] + m.N["z"], m.N["u"])), A_xz)),
            )
        )

        self.B_s = np.vstack((G @ B_xz, B_xz))

    def linearize_aim(self):
        """
        Linearize the model using the Anderson-Moore AIM algorithm.

        This method constructs the structural form matrices with leads and lags,
        then uses the AIM algorithm to solve for the reduced form.
        """
        m = self.model

        # Construct lag block of structural form
        H_lag = np.vstack(
            (
                np.zeros((m.N["u"], m.N["u"] + m.N["x"] + m.N["z"])),
                -np.hstack(
                    tuple(m.derivatives["transition"][var] for var in ["u", "x", "z"])
                ),
                np.hstack((np.zeros((m.N["z"], m.N["u"] + m.N["x"])), -self.Phi)),
            )
        )

        f_E = m.derivatives["optimality"]["E"]

        # Construct current period block
        H_t = np.vstack(
            (
                np.hstack(
                    tuple(
                        m.derivatives["optimality"][var]
                        + f_E @ m.derivatives["expectations"][var]
                        for var in ["u", "x", "z"]
                    )
                ),
                np.hstack(
                    (
                        np.zeros((m.N["x"], m.N["u"])),
                        np.eye(m.N["x"]),
                        np.zeros((m.N["x"], m.N["z"])),
                    )
                ),
                np.hstack(
                    (
                        np.zeros((m.N["z"], m.N["u"] + m.N["x"])),
                        np.eye(m.N["z"]),
                    )
                ),
            )
        )

        # Construct lead block
        H_new = np.vstack(
            (
                np.hstack(
                    tuple(
                        f_E @ m.derivatives["expectations"][var + "_new"]
                        for var in ["u", "x", "z"]
                    )
                ),
                np.zeros((m.N["x"] + m.N["z"], m.N["u"] + m.N["x"] + m.N["z"])),
            )
        )

        H_aim = np.hstack((H_lag, H_t, H_new))

        aim_obj = perturb.solve_aim(H_aim, nlead=1)

        self.A_s = aim_obj.B
        self.B_s = np.zeros((m.A_s.shape[0], self.impact_matrix.shape[1]))

    def simulate(self, Nt, s_init=None, shocks=None):
        """
        Simulate linearized model dynamics.

        Parameters
        ----------
        Nt : int
            Number of time periods to simulate.
        s_init : array_like, optional
            Initial state vector. If None, uses steady state.
        shocks : array_like, optional
            Matrix of shocks (Nt x Nshock). If None, uses zero shocks.

        Returns
        -------
        array_like
            Simulated state paths (Nt x Ns).
        """
        if s_init is None:
            s_init = self.model.get_s_steady()

        if shocks is None:
            shocks = np.zeros((Nt, self.B.shape[1]))

        Ns = self.A.shape[0]
        s_sim = np.zeros((Nt, Ns))

        s_t = s_init[:, np.newaxis]
        for tt in range(Nt):
            s_t = self.A @ s_t + self.B @ shocks[tt, :][:, np.newaxis]
            s_sim[tt, :] = s_t.ravel()

        return s_sim

    def compute_irfs(self, Nt_irf):
        """
        Compute impulse response functions for all shocks.

        Parameters
        ----------
        Nt_irf : int
            Number of periods for IRF horizon.

        Returns
        -------
        dict[str, IrfResult]
            Dictionary mapping shock names to IrfResult objects. Each IrfResult
            contains the impulse responses for all variables (UX, Z, Y) to that
            specific shock.

        Notes
        -----
        This method also stores the full IRF tensor in ``self.irfs`` for
        backward compatibility, with shape ``(Nshock, Nt_irf, Ns)`` where
        Ns = N_u + N_x + N_z + N_intermediate + N_E + N_expectations_vars.

        The intermediate variables (Y) are computed using the existing
        ``compute_linear_intermediates`` function from ``solvers.linear``.
        """
        from ..solvers.linear import compute_linear_intermediates
        from ..solvers.results import IrfResult

        Ns = self.A.shape[0]
        Nshock = self.B.shape[1]
        m = self.model

        # Store full IRF tensor for backward compatibility
        self.irfs = np.zeros((Nshock, Nt_irf, Ns))

        Psi = self.B.copy()
        for tt in range(Nt_irf):
            self.irfs[:, tt, :] = Psi.T
            if tt < Nt_irf - 1:
                Psi = self.A @ Psi

        # Extract dimensions
        N_u = m.N["u"]
        N_x = m.N["x"]
        N_z = m.N["z"]
        N_ux = N_u + N_x

        # Build IrfResult for each shock
        irf_dict = {}
        shock_names = m.exog_list

        for i_shock, shock_name in enumerate(shock_names):
            # Extract the IRF for this shock
            # self.irfs has shape (Nshock, Nt_irf, Ns)
            # where Ns includes [u; x; z; intermediates; E; expectations_vars]
            irf_full = self.irfs[i_shock, :, :]  # Shape: (Nt_irf, Ns)

            # Extract UX (control and state variables)
            UX = irf_full[:, :N_ux]  # Shape: (Nt_irf, N_ux)

            # Extract Z (exogenous variables)
            Z = irf_full[:, N_ux : N_ux + N_z]  # Shape: (Nt_irf, N_z)

            # Compute Y (intermediate variables) in deviation form
            # UX and Z from IRFs are already deviations from steady state
            Y = compute_linear_intermediates(m, UX, Z, deviations=True)

            # Get variable names
            var_names = m.var_lists["u"] + m.var_lists["x"]
            exog_names = m.exog_list
            y_names = m.linear_y_names() if hasattr(m, "linear_y_names") else []

            # Create IrfResult for this shock
            irf_result = IrfResult(
                UX=UX,
                Z=Z,
                Y=Y,
                model_label=m.label,
                var_names=var_names,
                exog_names=exog_names,
                y_names=y_names,
                shock_name=shock_name,
                shock_size=1.0,
            )

            irf_dict[shock_name] = irf_result

        # Cache the irf_dict for saving
        self._irf_dict = irf_dict

        return irf_dict

    def as_dict(self, include_irfs: bool = True):
        """
        Export linearized model matrices as a dictionary.

        Parameters
        ----------
        include_irfs : bool, default True
            If True and IRFs have been computed, include them in the output.

        Returns
        -------
        dict
            Dictionary containing all linearized model matrices and optionally IRFs.
        """
        result = {
            "A": self.A,
            "B": self.B,
            "A_s": self.A_s,
            "B_s": self.B_s,
            "G_x": self.G_x,
            "G_z": self.G_z,
            "H_x": self.H_x,
            "H_z": self.H_z,
            "Phi": self.Phi,
            "impact_matrix": self.impact_matrix,
        }
        if include_irfs and self.irfs is not None:
            result["irfs"] = self.irfs
        return result

    def save_irfs(
        self,
        filepath: Optional[str | Path] = None,
        *,
        format: str = "npz",
        include_matrices: bool = False,
        overwrite: bool = False,
        timestamp: bool = False,
    ) -> Path:
        """
        Save impulse response functions to a file.

        Parameters
        ----------
        filepath : str or Path, optional
            Path to save the IRFs. If None, uses defaults based on model label
            and settings.
        format : str, default "npz"
            Output format. Supported: 'npz', 'csv', 'json'.
        include_matrices : bool, default False
            If True, also save the linearization matrices (A, B, G_x, G_z, H_x, H_z).
        overwrite : bool, default False
            If False and file exists, raise FileExistsError.
        timestamp : bool, default False
            If True and filepath is None, append timestamp to filename.

        Returns
        -------
        Path
            The path to the saved file.

        Raises
        ------
        RuntimeError
            If no IRFs have been computed.
        FileExistsError
            If file exists and overwrite is False.
        """
        from ..io import resolve_output_path, save_results

        if self.irfs is None:
            raise RuntimeError("No IRFs have been computed. Call compute_irfs() first.")

        # Get model label from parent model
        model_label = getattr(self.model, "label", "_default")

        # Determine file extension based on format
        suffix_map = {"npz": ".npz", "json": ".json", "csv": ".csv"}
        suffix = suffix_map.get(format, ".npz")

        # Resolve output path
        path = resolve_output_path(
            filepath,
            result_type="irfs",
            model_label=model_label,
            timestamp=timestamp,
            suffix=suffix,
        )

        # Prepare data - save IrfResult objects with Y data
        # Compute IRFs if not already in IrfResult format
        if not hasattr(self, "_irf_dict") or self._irf_dict is None:
            # Need to compute IrfResults with Y data
            from ..solvers.linear import compute_linear_intermediates
            from ..solvers.results import IrfResult

            N_u = self.model.N["u"]
            N_x = self.model.N["x"]
            N_z = self.model.N["z"]
            N_ux = N_u + N_x
            shock_names = self.model.exog_list
            ux_names = self.model.var_lists["u"] + self.model.var_lists["x"]
            exog_names = self.model.exog_list
            y_names = self.model.var_lists.get("intermediate", [])

            self._irf_dict = {}
            for i_shock, shock_name in enumerate(shock_names):
                irf_full = self.irfs[i_shock, :, :]
                UX = irf_full[:, :N_ux]
                Z = irf_full[:, N_ux : N_ux + N_z]
                # IRFs are in deviation form
                Y = compute_linear_intermediates(self.model, UX, Z, deviations=True)

                self._irf_dict[shock_name] = IrfResult(
                    UX=UX,
                    Z=Z,
                    Y=Y,
                    model_label=model_label,
                    var_names=ux_names,
                    exog_names=exog_names,
                    y_names=y_names,
                    shock_name=shock_name,
                    shock_size=1.0,
                )

        # Save each IrfResult's data separately
        data = {"irfs": self.irfs}  # Keep for backward compatibility

        for shock_name, irf_result in self._irf_dict.items():
            data[f"UX_{shock_name}"] = irf_result.UX
            data[f"Z_{shock_name}"] = irf_result.Z
            if irf_result.Y is not None:
                data[f"Y_{shock_name}"] = irf_result.Y

        if include_matrices:
            data.update(self.as_dict(include_irfs=False))

        # Prepare metadata
        shock_names = getattr(self.model, "exog_list", [])
        var_names = getattr(self.model, "all_vars", [])
        ux_names = self.model.var_lists["u"] + self.model.var_lists["x"]
        exog_names = self.model.exog_list
        y_names = self.model.var_lists.get("intermediate", [])

        metadata = {
            "model_label": model_label,
            "shock_names": shock_names,
            "var_names": var_names,
            "ux_names": ux_names,
            "exog_names": exog_names,
            "y_names": y_names,
            "n_ux": len(ux_names),
            "n_z": len(exog_names),
            "irfs_shape": list(self.irfs.shape),
        }

        # Save
        return save_results(
            data,
            path,
            format=format,
            metadata=metadata,
            overwrite=overwrite,
        )
