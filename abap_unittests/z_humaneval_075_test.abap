CLASS ltcl_test_z_humaneval_075 DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_5 FOR TESTING,
      test_case_30 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_10 FOR TESTING,
      test_case_125 FOR TESTING,
      test_case_105 FOR TESTING,
      test_case_126 FOR TESTING,
      test_case_729 FOR TESTING,
      test_case_891 FOR TESTING,
      test_case_1001 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_z_humaneval_075 IMPLEMENTATION.

  METHOD test_case_5.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_075=>is_multiply_prime( 5 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_30.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_075=>is_multiply_prime( 30 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_8.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_075=>is_multiply_prime( 8 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_10.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_075=>is_multiply_prime( 10 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_125.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_075=>is_multiply_prime( 125 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_105.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_075=>is_multiply_prime( 105 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_126.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_075=>is_multiply_prime( 126 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_729.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_075=>is_multiply_prime( 729 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_891.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_075=>is_multiply_prime( 891 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_1001.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_075=>is_multiply_prime( 1001 )
      exp = abap_true ).
  ENDMETHOD.

ENDCLASS.