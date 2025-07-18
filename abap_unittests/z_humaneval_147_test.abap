CLASS ltcl_test_z_humaneval_147 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_z_humaneval_147 IMPLEMENTATION.
  METHOD test_case_1.
    DATA: result TYPE i.
    result = z_humaneval_147=>get_max_triples( 5 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 1
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: result TYPE i.
    result = z_humaneval_147=>get_max_triples( 6 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 4
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: result TYPE i.
    result = z_humaneval_147=>get_max_triples( 10 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 36
    ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: result TYPE i.
    result = z_humaneval_147=>get_max_triples( 100 ).
    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 53361
    ).
  ENDMETHOD.
ENDCLASS.