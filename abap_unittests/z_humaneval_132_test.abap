CLASS test_z_humaneval_132 DEFINITION
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
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING,
      test_case_10 FOR TESTING,
      test_case_11 FOR TESTING,
      test_case_12 FOR TESTING,
      test_case_13 FOR TESTING,
      test_case_14 FOR TESTING.

ENDCLASS.

CLASS test_z_humaneval_132 IMPLEMENTATION.

  METHOD test_case_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '[[]]' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '[]]]]]]][[[[[]' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_3.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '[][]' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '[]' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_5.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '[[[[]]]]' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_6.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '[]]]]]]]]]]' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_7.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '[][][[]]' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_8.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '[[]' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_9.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '[]]' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_10.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '[[]][[' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_11.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '[[][]]' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_12.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_13.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( '[[[[[[[[' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_14.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_132=>is_nested( ']]]]]]]]' )
      exp = abap_false ).
  ENDMETHOD.

ENDCLASS.