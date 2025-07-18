CLASS lhc_z_humaneval_025 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS test_factorize_2 FOR TESTING.
    METHODS test_factorize_4 FOR TESTING.
    METHODS test_factorize_8 FOR TESTING.
    METHODS test_factorize_3_times_19 FOR TESTING.
    METHODS test_factorize_3_19_3_19 FOR TESTING.
    METHODS test_factorize_3_19_3_19_3_19 FOR TESTING.
    METHODS test_factorize_3_19_19_19 FOR TESTING.
    METHODS test_factorize_3_2_3 FOR TESTING.
ENDCLASS.

CLASS lhc_z_humaneval_025 IMPLEMENTATION.

  METHOD test_factorize_2.
    DATA(result) = z_humaneval_025=>factorize( 2 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 2 ) ) ).
  ENDMETHOD.

  METHOD test_factorize_4.
    DATA(result) = z_humaneval_025=>factorize( 4 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 2 ) ( 2 ) ) ).
  ENDMETHOD.

  METHOD test_factorize_8.
    DATA(result) = z_humaneval_025=>factorize( 8 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 2 ) ( 2 ) ( 2 ) ) ).
  ENDMETHOD.

  METHOD test_factorize_3_times_19.
    DATA(result) = z_humaneval_025=>factorize( 3 * 19 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 3 ) ( 19 ) ) ).
  ENDMETHOD.

  METHOD test_factorize_3_19_3_19.
    DATA(result) = z_humaneval_025=>factorize( 3 * 19 * 3 * 19 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 3 ) ( 3 ) ( 19 ) ( 19 ) ) ).
  ENDMETHOD.

  METHOD test_factorize_3_19_3_19_3_19.
    DATA(result) = z_humaneval_025=>factorize( 3 * 19 * 3 * 19 * 3 * 19 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 3 ) ( 3 ) ( 3 ) ( 19 ) ( 19 ) ( 19 ) ) ).
  ENDMETHOD.

  METHOD test_factorize_3_19_19_19.
    DATA(result) = z_humaneval_025=>factorize( 3 * 19 * 19 * 19 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 3 ) ( 19 ) ( 19 ) ( 19 ) ) ).
  ENDMETHOD.

  METHOD test_factorize_3_2_3.
    DATA(result) = z_humaneval_025=>factorize( 3 * 2 * 3 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = VALUE int4_table( ( 2 ) ( 3 ) ( 3 ) ) ).
  ENDMETHOD.

ENDCLASS.