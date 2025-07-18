CLASS ltcl_test_right_angle_triangle DEFINITION
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
      test_case_11 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_right_angle_triangle IMPLEMENTATION.

  METHOD test_case_1.
    cl_abap_unit_assert=>assert_true(
      act = z_humaneval_157=>right_angle_triangle( a = 3 b = 4 c = 5 )
      msg = 'Test case 1 failed'
    ).
  ENDMETHOD.

  METHOD test_case_2.
    cl_abap_unit_assert=>assert_false(
      act = z_humaneval_157=>right_angle_triangle( a = 1 b = 2 c = 3 )
      msg = 'Test case 2 failed'
    ).
  ENDMETHOD.

  METHOD test_case_3.
    cl_abap_unit_assert=>assert_true(
      act = z_humaneval_157=>right_angle_triangle( a = 10 b = 6 c = 8 )
      msg = 'Test case 3 failed'
    ).
  ENDMETHOD.

  METHOD test_case_4.
    cl_abap_unit_assert=>assert_false(
      act = z_humaneval_157=>right_angle_triangle( a = 2 b = 2 c = 2 )
      msg = 'Test case 4 failed'
    ).
  ENDMETHOD.

  METHOD test_case_5.
    cl_abap_unit_assert=>assert_true(
      act = z_humaneval_157=>right_angle_triangle( a = 7 b = 24 c = 25 )
      msg = 'Test case 5 failed'
    ).
  ENDMETHOD.

  METHOD test_case_6.
    cl_abap_unit_assert=>assert_false(
      act = z_humaneval_157=>right_angle_triangle( a = 10 b = 5 c = 7 )
      msg = 'Test case 6 failed'
    ).
  ENDMETHOD.

  METHOD test_case_7.
    cl_abap_unit_assert=>assert_true(
      act = z_humaneval_157=>right_angle_triangle( a = 5 b = 12 c = 13 )
      msg = 'Test case 7 failed'
    ).
  ENDMETHOD.

  METHOD test_case_8.
    cl_abap_unit_assert=>assert_true(
      act = z_humaneval_157=>right_angle_triangle( a = 15 b = 8 c = 17 )
      msg = 'Test case 8 failed'
    ).
  ENDMETHOD.

  METHOD test_case_9.
    cl_abap_unit_assert=>assert_true(
      act = z_humaneval_157=>right_angle_triangle( a = 48 b = 55 c = 73 )
      msg = 'Test case 9 failed'
    ).
  ENDMETHOD.

  METHOD test_case_10.
    cl_abap_unit_assert=>assert_false(
      act = z_humaneval_157=>right_angle_triangle( a = 1 b = 1 c = 1 )
      msg = 'Test case 10 failed'
    ).
  ENDMETHOD.

  METHOD test_case_11.
    cl_abap_unit_assert=>assert_false(
      act = z_humaneval_157=>right_angle_triangle( a = 2 b = 2 c = 10 )
      msg = 'Test case 11 failed'
    ).
  ENDMETHOD.

ENDCLASS.