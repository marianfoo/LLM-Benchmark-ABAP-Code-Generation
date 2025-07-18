CLASS ltcl_test_z_humaneval_161 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_z_humaneval_161 IMPLEMENTATION.

  METHOD test_case_1.
    DATA: result TYPE string.
    result = z_humaneval_161=>solve( 'AsDf' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'aSdF' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: result TYPE string.
    result = z_humaneval_161=>solve( '1234' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '4321' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: result TYPE string.
    result = z_humaneval_161=>solve( 'ab' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'AB' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: result TYPE string.
    result = z_humaneval_161=>solve( '#a@C' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '#A@c' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: result TYPE string.
    result = z_humaneval_161=>solve( '#AsdfW^45' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '#aSDFw^45' ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: result TYPE string.
    result = z_humaneval_161=>solve( '#6@2' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '2@6#' ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: result TYPE string.
    result = z_humaneval_161=>solve( '#$a^D' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '#$A^d' ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: result TYPE string.
    result = z_humaneval_161=>solve( '#ccc' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '#CCC' ).
  ENDMETHOD.

ENDCLASS.