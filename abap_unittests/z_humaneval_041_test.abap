CLASS ltcl_test_car_race_collision DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_car_race_collision IMPLEMENTATION.
  METHOD test_case_1.
    DATA(result) = z_humaneval_041=>car_race_collision( 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 4 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA(result) = z_humaneval_041=>car_race_collision( 3 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 9 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA(result) = z_humaneval_041=>car_race_collision( 4 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 16 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA(result) = z_humaneval_041=>car_race_collision( 8 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 64 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA(result) = z_humaneval_041=>car_race_collision( 10 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 100 ).
  ENDMETHOD.
ENDCLASS.