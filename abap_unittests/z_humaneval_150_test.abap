CLASS ltcl_test_z_humaneval_150 DEFINITION
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
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING,
      test_case_10 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_z_humaneval_150 IMPLEMENTATION.

  METHOD test_case_1.
    DATA(result) = z_humaneval_150=>x_or_y( n = 7 x = 34 y = 12 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 34 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA(result) = z_humaneval_150=>x_or_y( n = 15 x = 8 y = 5 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 5 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA(result) = z_humaneval_150=>x_or_y( n = 3 x = 33 y = 5212 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 33 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA(result) = z_humaneval_150=>x_or_y( n = 1259 x = 3 y = 52 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 3 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA(result) = z_humaneval_150=>x_or_y( n = 7919 x = -1 y = 12 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA(result) = z_humaneval_150=>x_or_y( n = 3609 x = 1245 y = 583 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 583 ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA(result) = z_humaneval_150=>x_or_y( n = 91 x = 56 y = 129 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 129 ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA(result) = z_humaneval_150=>x_or_y( n = 6 x = 34 y = 1234 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1234 ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA(result) = z_humaneval_150=>x_or_y( n = 1 x = 2 y = 0 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_10.
    DATA(result) = z_humaneval_150=>x_or_y( n = 2 x = 2 y = 0 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.

ENDCLASS.