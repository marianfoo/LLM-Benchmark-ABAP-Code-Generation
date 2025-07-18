CLASS ltcl_test_choose_num DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_choose_num IMPLEMENTATION.

  METHOD test_case_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_102=>choose_num( x = 12 y = 15 )
      exp = 14 ).
  ENDMETHOD.

  METHOD test_case_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_102=>choose_num( x = 13 y = 12 )
      exp = -1 ).
  ENDMETHOD.

  METHOD test_case_3.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_102=>choose_num( x = 33 y = 12354 )
      exp = 12354 ).
  ENDMETHOD.

  METHOD test_case_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_102=>choose_num( x = 5234 y = 5233 )
      exp = -1 ).
  ENDMETHOD.

  METHOD test_case_5.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_102=>choose_num( x = 6 y = 29 )
      exp = 28 ).
  ENDMETHOD.

  METHOD test_case_6.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_102=>choose_num( x = 27 y = 10 )
      exp = -1 ).
  ENDMETHOD.

  METHOD test_case_7.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_102=>choose_num( x = 7 y = 7 )
      exp = -1 ).
  ENDMETHOD.

  METHOD test_case_8.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_102=>choose_num( x = 546 y = 546 )
      exp = 546 ).
  ENDMETHOD.

ENDCLASS.